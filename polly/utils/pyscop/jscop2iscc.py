#!/usr/bin/python
import argparse, isl, os
import json

def printDomain(scop):

  domain = isl.USet('{}')

  for statement in scop['statements']:
    domain = domain.union(isl.USet(statement['domain']))

  print "D :=",
  print str(domain) + ";"

def printAccesses(scop):

  read = isl.UMap('{}')

  for statement in scop['statements']:
    for access in statement['accesses']:
      if access['kind'] == 'read':
        read = read.union(isl.UMap(access['relation']))

  print "R :=",
  print str(read) + ";"

  write = isl.UMap('{}')

  for statement in scop['statements']:
    for access in statement['accesses']:
      if access['kind'] == 'write':
        write = write.union(isl.UMap(access['relation']))

  print "W :=",
  print str(write) + ";"

def printSchedule(scop):

  schedule = isl.UMap('{}')

  for statement in scop['statements']:
    schedule = schedule.union(isl.UMap(statement['schedule']))

  print "S :=",
  print str(schedule) + ";"

def __main__():
  description = 'Translate JSCoP into iscc input'
  parser = argparse.ArgumentParser(description)
  parser.add_argument('inputFile', metavar='N', type=file,
                      help='The JSCoP file')

  args = parser.parse_args()
  inputFile = args.inputFile
  scop = json.load(inputFile)

  printDomain(scop)
  printAccesses(scop)
  printSchedule(scop)

  print 'R := R * D;'
  print 'W := W * D;'
  print 'Dep := (last W before R under S)[0];'
  print 'schedule D respecting Dep minimizing Dep;'


__main__()

