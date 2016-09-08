#!/usr/bin/env python

""" This runs netstat on a local or remote server. It calculates some simple
statistical information on the number of external inet connections. It groups
by IP address. This can be used to detect if one IP address is taking up an
excessive number of connections. It can also send an email alert if a given IP
address exceeds a threshold between runs of the script. This script can be used
as a drop-in Munin plugin or it can be used stand-alone from cron. I used this
on a busy web server that would sometimes get hit with denial of service
attacks. This made it easy to see if a script was opening many multiple
connections. A typical browser would open fewer than 10 connections at once. A
script might open over 100 simultaneous connections.

./topip.py [-s server_hostname] [-u username] [-p password] {-a from_addr,to_addr} {-n N} {-v} {--ipv6}

    -s : hostname of the remote server to login to.
    -u : username to user for login.
    -p : password to user for login.
    -n : print stddev for the the number of the top 'N' ipaddresses.
    -v : verbose - print stats and list of top ipaddresses.
    -a : send alert if stddev goes over 20.
    -l : to log message to /var/log/topip.log
    --ipv6 : this parses netstat output that includes ipv6 format.
        Note that this actually only works with ipv4 addresses, but for versions of
        netstat that print in ipv6 format.
    --stdev=N : Where N is an integer. This sets the trigger point for alerts and logs.
        Default is to trigger if max value is above 5 standard deviations.

Example:

    This will print stats for the top IP addresses connected to the given host:

        ./topip.py -s www.example.com -u mylogin -p mypassword -n 10 -v

    This will send an alert email if the maxip goes over the stddev trigger value and
    the the current top ip is the same as the last top ip (/tmp/topip.last):

        ./topip.py -s www.example.com -u mylogin -p mypassword -n 10 -v -a alert@example.com,user@example.com

    This will print the connection stats for the localhost in Munin format:

        ./topip.py

Noah Spurrier

$Id: topip.py 489 2007-11-28 23:40:34Z noah $
"""

import pexpect
import pxssh  # See http://pexpect.sourceforge.net/
import os
import sys
import time
import re
import getopt
import pickle
import getpass
import smtplib
import traceback
from pprint import pprint

TOPIP_LOG_FILE = '/var/log/topip.log'
TOPIP_LAST_RUN_STATS = '/var/run/topip.last'


def exit_with_usage():

    print globals()['__doc__']
    os._exit(1)


def stats(r):
    """This returns a dict of the median, average, standard deviation, min and max of the given sequence.

    >>> from topip import stats
    >>> print stats([5,6,8,9])
    {'med': 8, 'max': 9, 'avg': 7.0, 'stddev': 1.5811388300841898, 'min': 5}
    >>> print stats([1000,1006,1008,1014])
    {'med': 1008, 'max': 1014, 'avg': 1007.0, 'stddev': 5.0, 'min': 1000}
    >>> print stats([1,3,4,5,18,16,4,3,3,5,13])
    {'med': 4, 'max': 18, 'avg': 6.8181818181818183, 'stddev': 5.6216817577237475, 'min': 1}
    >>> print stats([1,3,4,5,18,16,4,3,3,5,13,14,5,6,7,8,7,6,6,7,5,6,4,14,7])
    {'med': 6, 'max': 18, 'avg': 7.0800000000000001, 'stddev': 4.3259218670706474, 'min': 1}
    """

    total = sum(r)
    avg = float(total) / float(len(r))
    sdsq = sum([(i - avg)**2 for i in r])
    s = sorted(r)
    return dict(zip(['med', 'avg', 'stddev', 'min', 'max'],
                    (s[len(s) // 2], avg, (sdsq / len(r))**.5, min(r), max(r))))


def send_alert(message, subject, addr_from, addr_to, smtp_server='localhost'):
    """This sends an email alert.
    """

    message = 'From: %s\r\nTo: %s\r\nSubject: %s\r\n\r\n' % (
        addr_from, addr_to, subject) + message
    server = smtplib.SMTP(smtp_server)
    server.sendmail(addr_from, addr_to, message)
    server.quit()


def main():

    ######################################################################
    # Parse the options, arguments, etc.
    ######################################################################
    try:
        optlist, args = getopt.getopt(
            sys.argv[
                1:], 'h?valqs:u:p:n:', [
                'help', 'h', '?', 'ipv6', 'stddev='])
    except Exception as e:
        print str(e)
        exit_with_usage()
    options = dict(optlist)

    munin_flag = False
    if len(args) > 0:
        if args[0] == 'config':
            print 'graph_title Netstat Connections per IP'
            print 'graph_vlabel Socket connections per IP'
            print 'connections_max.label max'
            print 'connections_max.info Maximum number of connections per IP'
            print 'connections_avg.label avg'
            print 'connections_avg.info Average number of connections per IP'
            print 'connections_stddev.label stddev'
            print 'connections_stddev.info Standard deviation'
            return 0
        elif args[0] != '':
            print args, len(args)
            return 0
            exit_with_usage()
    if [elem for elem in options if elem in [
            '-h', '--h', '-?', '--?', '--help']]:
        print 'Help:'
        exit_with_usage()
    if '-s' in options:
        hostname = options['-s']
    else:
        # if host was not specified then assume localhost munin plugin.
        munin_flag = True
        hostname = 'localhost'
    # If localhost then don't ask for username/password.
    if hostname != 'localhost' and hostname != '127.0.0.1':
        if '-u' in options:
            username = options['-u']
        else:
            username = raw_input('username: ')
        if '-p' in options:
            password = options['-p']
        else:
            password = getpass.getpass('password: ')
    else:
        use_localhost = True

    if '-l' in options:
        log_flag = True
    else:
        log_flag = False
    if '-n' in options:
        average_n = int(options['-n'])
    else:
        average_n = None
    if '-v' in options:
        verbose = True
    else:
        verbose = False
    if '-a' in options:
        alert_flag = True
        (alert_addr_from, alert_addr_to) = tuple(options['-a'].split(','))
    else:
        alert_flag = False
    if '--ipv6' in options:
        ipv6_flag = True
    else:
        ipv6_flag = False
    if '--stddev' in options:
        stddev_trigger = float(options['--stddev'])
    else:
        stddev_trigger = 5

    if ipv6_flag:
        netstat_pattern = '(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+::ffff:(\S+):(\S+)\s+.*?\r'
    else:
        netstat_pattern = '(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(?:::ffff:)*(\S+):(\S+)\s+.*?\r'
        #netstat_pattern = '(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+):(\S+)\s+.*?\r'

    # run netstat (either locally or via SSH).
    if use_localhost:
        p = pexpect.spawn('netstat -n -t')
        PROMPT = pexpect.TIMEOUT
    else:
        p = pxssh.pxssh()
        p.login(hostname, username, password)
        p.sendline('netstat -n -t')
        PROMPT = p.PROMPT

    # loop through each matching netstat_pattern and put the ip address in the
    # list.
    ip_list = {}
    try:
        while True:
            i = p.expect([PROMPT, netstat_pattern])
            if i == 0:
                break
            k = p.match.groups()[4]
            if k in ip_list:
                ip_list[k] = ip_list[k] + 1
            else:
                ip_list[k] = 1
    except:
        pass

    # remove a few common, uninteresting addresses from the dictionary.
    ip_list = dict([(key, value)
                    for key, value in ip_list.items() if '192.168.' not in key])
    ip_list = dict([(key, value)
                    for key, value in ip_list.items() if '127.0.0.1' not in key])

    # sort dict by value (count)
    #ip_list = sorted(ip_list.iteritems(),lambda x,y:cmp(x[1], y[1]),reverse=True)
    ip_list = ip_list.items()
    if len(ip_list) < 1:
        if verbose:
            print 'Warning: no networks connections worth looking at.'
        return 0
    ip_list.sort(lambda x, y: cmp(y[1], x[1]))

    # generate some stats for the ip addresses found.
    if average_n <= 1:
        average_n = None
    # The * unary operator treats the list elements as arguments
    s = stats(zip(*ip_list[0:average_n])[1])
    s['maxip'] = ip_list[0]

    # print munin-style or verbose results for the stats.
    if munin_flag:
        print 'connections_max.value', s['max']
        print 'connections_avg.value', s['avg']
        print 'connections_stddev.value', s['stddev']
        return 0
    if verbose:
        pprint(s)
        print
        pprint(ip_list[0:average_n])

    # load the stats from the last run.
    try:
        last_stats = pickle.load(file(TOPIP_LAST_RUN_STATS))
    except:
        last_stats = {'maxip': None}

    if s['maxip'][1] > (
        s['stddev'] *
            stddev_trigger) and s['maxip'] == last_stats['maxip']:
        if verbose:
            print 'The maxip has been above trigger for two consecutive samples.'
        if alert_flag:
            if verbose:
                print 'SENDING ALERT EMAIL'
            send_alert(
                str(s),
                'ALERT on %s' %
                hostname,
                alert_addr_from,
                alert_addr_to)
        if log_flag:
            if verbose:
                print 'LOGGING THIS EVENT'
            fout = file(TOPIP_LOG_FILE, 'a')
            #dts = time.strftime('%Y:%m:%d:%H:%M:%S', time.localtime())
            dts = time.asctime()
            fout.write('%s - %d connections from %s\n' %
                       (dts, s['maxip'][1], str(s['maxip'][0])))
            fout.close()

    # save state to TOPIP_LAST_RUN_STATS
    try:
        pickle.dump(s, file(TOPIP_LAST_RUN_STATS, 'w'))
        os.chmod(TOPIP_LAST_RUN_STATS, 0o664)
    except:
        pass
    # p.logout()

if __name__ == '__main__':
    try:
        main()
        sys.exit(0)
    except SystemExit as e:
        raise e
    except Exception as e:
        print str(e)
        traceback.print_exc()
        os._exit(1)
