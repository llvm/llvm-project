#!/usr/bin/env python
import argparse, os
import json

def getDomains(scop):
  statements = scop['statements'];
  numStatements = len(statements)

  output = "%s\n\n" % str(numStatements)

  for statement in scop['statements']:
    output += "%s\n\n" % statement['domain']
    output += "0  0  0               # for future options\n\n"


  return output

def getSchedules(scop):
  statements = scop['statements'];
  numStatements = len(statements)

  output = "%s\n\n" % str(numStatements)

  for statement in scop['statements']:
    output += "%s\n\n" % statement['schedule']

  return output

def writeCloog(scop):
  template = """
# ---------------------- CONTEXT ----------------------
c # language is C

# Context (no constraints on two parameters)
%s

0 # We do not want to set manually the parameter names

# --------------------- STATEMENTS --------------------
%s

0 # We do not want to set manually the iterator names

# --------------------- SCATTERING --------------------
%s

0 # We do not want to set manually the schedule dimension names
"""

  context = scop['context']
  domains = getDomains(scop)
  schedules = getSchedules(scop)
  print template % (context, domains, schedules)

def __main__():
  description = 'Translate JSCoP into iscc input'
  parser = argparse.ArgumentParser(description)
  parser.add_argument('inputFile', metavar='N', type=file,
                      help='The JSCoP file')

  args = parser.parse_args()
  inputFile = args.inputFile
  scop = json.load(inputFile)

  writeCloog(scop)

__main__()

