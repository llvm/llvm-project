#!/usr/bin/python
import re, string, sys, os, time, math

DEBUG = 0

(tp, exp) = ('compile', 'exec')

def parse(file):
  f = open(file, 'r')
  d = f.read()
  
  # Cleanup weird stuff
  d = re.sub(r',\d+:\d', '', d)

  r = re.findall(r'TEST-(PASS|FAIL|RESULT.*?):\s+(.*?)\s+(.*?)\r*\n', d)

  test = {}
  fname = ''
  for t in r:
    if DEBUG:
      print t

    if t[0] == 'PASS' or t[0] == 'FAIL' :
      tmp = t[2].split('llvm-test/')
      
      if DEBUG:
        print tmp

      if len(tmp) == 2:
        fname = tmp[1].strip('\r\n')
      else:
        fname = tmp[0].strip('\r\n')

      if not test.has_key(fname):
        test[fname] = {}

      test[fname][t[1] + ' state'] = t[0]
      test[fname][t[1] + ' time'] = float('nan')
    else :
      try:
        n = t[0].split('RESULT-')[1]

        if DEBUG:
          print "n == ", n;
        
        if n == 'compile-success':
          test[fname]['compile time'] = float(t[2].split('program')[1].strip('\r\n'))

        elif n == 'exec-success':
          test[fname]['exec time'] = float(t[2].split('program')[1].strip('\r\n'))
          if DEBUG:
            print test[fname][string.replace(n, '-success', '')]

        else :
          # print "ERROR!"
          sys.exit(1)

      except:
          continue

  return test

# Diff results and look for regressions.
def diffResults(d_old, d_new):

  for t in sorted(d_old.keys()) :
    if d_new.has_key(t):

      # Check if the test passed or failed.
      for x in ['compile state', 'compile time', 'exec state', 'exec time']:

        if not d_old[t].has_key(x) and not d_new[t].has_key(x):
          continue

        if d_old[t].has_key(x):
          if d_new[t].has_key(x):

            if d_old[t][x] == 'PASS':
              if d_new[t][x] != 'PASS':
                print t + " *** REGRESSION (" + x + " now fails)"
            else:
              if d_new[t][x] == 'PASS':
                print t + " * NEW PASS (" + x + " now fails)"

          else :
            print t + "*** REGRESSION (" + x + " now fails)"

        if x == 'compile state' or x == 'exec state':
          continue

        # For execution time, if there is no result it's a fail.
        if not d_old[t].has_key(x) and not d_new[t].has_key(x):
          continue
        elif not d_new[t].has_key(x):
          print t + " *** REGRESSION (" + x + ")"
        elif not d_old[t].has_key(x):
          print t + " * NEW PASS (" + x + ")"

        if math.isnan(d_old[t][x]) and math.isnan(d_new[t][x]):
          continue

        elif math.isnan(d_old[t][x]) and not math.isnan(d_new[t][x]):
          print t + " * NEW PASS (" + x + ")"

        elif not math.isnan(d_old[t][x]) and math.isnan(d_new[t][x]):
          print t + " *** REGRESSION (" + x + ")"

        if d_new[t][x] > d_old[t][x] and \
              (d_new[t][x] - d_old[t][x]) / d_new[t][x] > .05:
          print t + " *** REGRESSION (" + x + ")"

    else :
      print t + ": Removed from test-suite."

# Main
if len(sys.argv) < 3 :
  print 'Usage:', sys.argv[0], '<old log> <new log>'
  sys.exit(-1)

d_old = parse(sys.argv[1])
d_new = parse(sys.argv[2])

diffResults(d_old, d_new)
