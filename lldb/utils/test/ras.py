#!/usr/bin/env python

"""
Run the test suite and send the result as an email message.

The code for sending of the directory is copied from
http://docs.python.org/library/email-examples.html.
"""

import os
import sys
import shutil
import smtplib
# For guessing MIME type based on file name extension
import mimetypes

from optparse import OptionParser

from email import encoders
from email.message import Message
from email.mime.audio import MIMEAudio
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

def runTestsuite(testDir, sessDir, envs = None):
    """Run the testsuite and return a (summary, output) tuple."""
    os.chdir(testDir)

    for env in envs:
        list = env.split('=')
        var = list[0].strip()
        val = list[1].strip()
        print var + "=" + val
        os.environ[var] = val

    import shlex, subprocess

    command_line = "./dotest.py -w -s %s" % sessDir
    # Apply correct tokenization for subprocess.Popen().
    args = shlex.split(command_line)

    # Use subprocess module to spawn a new process.
    process = subprocess.Popen(args,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # Wait for subprocess to terminate.
    stdout, stderr = process.communicate()
    
    # This will be used as the subject line of our email about this test.
    cmd = "%s %s" % (' '.join(envs) if envs else "", command_line)

    return (cmd, stderr)


COMMASPACE = ', '

def main():
    parser = OptionParser(usage="""\
Run lldb test suite and send the results as a MIME message.

Usage: %prog [options]

Unless the -o option is given, the email is sent by forwarding to the specified
SMTP server, which then does the normal delivery process.
""")
    parser.add_option('-d', '--directory',
                      type='string', action='store',
                      dest='testDir',
                      help="""The LLDB test directory directly under the top dir.
                      Otherwise use the current directory.""")
    #
    # This is similar to TestBase.getRunSpec(self) from lldbtest.py.
    #
    parser.add_option('-e', '--environment',
                      type='string', action='append', metavar='ENVIRONMENT',
                      default=[], dest='environments',
                      help="""The environment setting as prefix to the test driver.
                      Example: -e 'CC=clang' -e 'ARCH=x86_64'""")
    parser.add_option('-m', '--mailserver',
                      type='string', action='store', metavar='MAILSERVER',
                      dest='mailserver',
                      help="""The outgoing SMTP server.""")
    parser.add_option('-o', '--output',
                      type='string', action='store', metavar='FILE',
                      help="""Print the composed message to FILE instead of
                      sending the message to the SMTP server.""")
    parser.add_option('-s', '--sender',
                      type='string', action='store', metavar='SENDER',
                      help='The value of the From: header (required)')
    parser.add_option('-r', '--recipient',
                      type='string', action='append', metavar='RECIPIENT',
                      default=[], dest='recipients',
                      help='A To: header value (at least one required)')
    opts, args = parser.parse_args()
    if not opts.sender or not opts.recipients:
        parser.print_help()
        sys.exit(1)
    testDir = opts.testDir
    if not testDir:
        testDir = '.'

    sessDir = 'tmp-lldb-session'
    if os.path.exists(sessDir):
        shutil.rmtree(sessDir)
    #print "environments:", opts.environments
    summary, output = runTestsuite(testDir, sessDir, opts.environments)

    # Create the enclosing (outer) message
    outer = MIMEMultipart()
    outer['Subject'] = summary
    outer['To'] = COMMASPACE.join(opts.recipients)
    outer['From'] = opts.sender
    outer.preamble = 'You will not see this in a MIME-aware mail reader.\n'

    # The sessDir contains all the session logs for failed/errored tests.
    # Attach them all if it exists!

    if not os.path.exists(sessDir):
        outer.attach(MIMEText(output, 'plain'))
    else:
        outer.attach(MIMEText("%s\n%s\n\n" % (output,
                                              "Session logs of test failures/errors:"),
                              'plain'))

    for filename in (os.listdir(sessDir) if os.path.exists(sessDir) else []):
        path = os.path.join(sessDir, filename)
        if not os.path.isfile(path):
            continue
        # Guess the content type based on the file's extension.  Encoding
        # will be ignored, although we should check for simple things like
        # gzip'd or compressed files.
        ctype, encoding = mimetypes.guess_type(path)
        if ctype is None or encoding is not None:
            # No guess could be made, or the file is encoded (compressed), so
            # use a generic bag-of-bits type.
            ctype = 'application/octet-stream'
        maintype, subtype = ctype.split('/', 1)
        if maintype == 'text':
            fp = open(path)
            # Note: we should handle calculating the charset
            msg = MIMEText(fp.read(), _subtype=subtype)
            fp.close()
        elif maintype == 'image':
            fp = open(path, 'rb')
            msg = MIMEImage(fp.read(), _subtype=subtype)
            fp.close()
        elif maintype == 'audio':
            fp = open(path, 'rb')
            msg = MIMEAudio(fp.read(), _subtype=subtype)
            fp.close()
        else:
            fp = open(path, 'rb')
            msg = MIMEBase(maintype, subtype)
            msg.set_payload(fp.read())
            fp.close()
            # Encode the payload using Base64
            encoders.encode_base64(msg)
        # Set the filename parameter
        msg.add_header('Content-Disposition', 'attachment', filename=filename)
        outer.attach(msg)

    # Now send or store the message
    composed = outer.as_string()
    if opts.output:
        fp = open(opts.output, 'w')
        fp.write(composed)
        fp.close()
    else:
        s = smtplib.SMTP(opts.mailserver)
        s.sendmail(opts.sender, opts.recipients, composed)
        s.quit()


if __name__ == '__main__':
    main()
