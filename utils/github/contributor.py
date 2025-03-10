#!/usr/bin/env python3

import argparse
import subprocess
import json
import sys
import re
import os
import time

from datetime import datetime
from functools import cache

contrib_database = {}
contrib_database_path = os.path.join(os.getcwd(),'contributors.json')
verbose = False
start_time = datetime.now()

def CreateArgParser():
  parser = argparse.ArgumentParser(prog='contributor', description='LLVM GitHub Organization Scripts')
  parser.add_argument('action', choices=['register', 'stats', 'print', 'query'], help='Action to perform')
  parser.add_argument('extra_args', nargs='+', default=[])
  parser.add_argument('--database', '-d', default=os.path.join(os.getcwd(),'contributors.json'), metavar='path', required=False, help='Path to contributor database')
  parser.add_argument('--verbose', '-v', required=False, action='store_true', help='Enable verbose logging')
  parser.add_argument('--filter', '-f', required=False, default='all', choices=['complete', 'noreply', 'missing', 'all'], help='Filter contributor database')
  return parser

@cache
def ParseArgs():
  parser = CreateArgParser()
  return parser.parse_args(sys.argv[1:])

def ElapsedTime():
  return str(datetime.now() - start_time)

def Checkpoint(msg=''):
  if not ParseArgs().verbose:
    return
  if len(msg) > 0:
    print('%s - Time elapsed %s' % (msg, ElapsedTime()))
  else:
    print('Time elapsed: %s' % ElapsedTime())

def InvokeAndDecode(cmd):
  for Attempt in range(10):
    try:
      status = subprocess.check_output(cmd)
      return json.loads(status)
    except:
      time.sleep(60)
  return None

def QueryOrgMembers():
  members = []
  page = 1
  while True:
    ghCommand = ['gh', 'api', '-H', 'Accept: application/vnd.github+json',
               '-H', 'X-GitHub-Api-Version: 2022-11-28', '/orgs/llvm/members?per_page=100&page=%d' % page]
    status = subprocess.check_output(ghCommand)
    new_members = json.loads(status)
    if len(new_members) == 0:
      Checkpoint('Finished org query')
      return members
    members.extend(new_members)
    page += 1

def QueryUser(user):
  ghCommand = ['gh', 'api', '-H', 'Accept: application/vnd.github+json',
               '-H', 'X-GitHub-Api-Version: 2022-11-28', '/users/%s' % user]
  return InvokeAndDecode(ghCommand)

def LookupLastCommit(user):
  ghCommand = ['gh', 'api', '-H', 'Accept: application/vnd.github+json',
               '-H' 'X-GitHub-Api-Version: 2022-11-28',
               '/repos/llvm/llvm-project/commits?author=%s&per_page=1' % user]
  return InvokeAndDecode(ghCommand)

def LoadContributorDatabase():
  contrib_database_path = ParseArgs().database
  if not os.path.exists(contrib_database_path):
    Checkpoint('Starting with empty contributor database (%s).' % contrib_database_path)
    return {}
  with open(contrib_database_path, 'r') as file:
    data = file.read()
    contrib_database = json.loads(data)
    if not contrib_database:
      Checkpoint('Initializing contributor database')
      return {}
    Checkpoint('Contributor database loaded %d entries.' % len(contrib_database))
    return contrib_database
  return {}

def WriteContributorDatabase(db):
  contrib_database_path = ParseArgs().database
  with open(contrib_database_path, 'w') as file:
    json.dump(db, file)
    Checkpoint('Saved database')

def GenerateUserProfile(login):
  Checkpoint('Generating user: %s' % login)
  user = {'login': login}
  userQuery = QueryUser(login)
  if userQuery and 'email' in userQuery and userQuery['email']:
    user['email'] = userQuery['email']
  else:
    commits = LookupLastCommit(login)
    if commits and len(commits) > 0:
      user['email'] = commits[0]['commit']['author']['email']
  return user

def RegisterContributors():
  contrib_database = LoadContributorDatabase()
  orgMembers = QueryOrgMembers()
  print('%d organization members identified' % len(orgMembers))
  processed = 0
  for member in orgMembers:
    if processed % 500 == 0:
      Checkpoint('Processed %d' % processed)
      WriteContributorDatabase(contrib_database)
    processed += 1
    # For now skip members that are already in the DB...
    if member['login'] in contrib_database:
      continue
    userData = GenerateUserProfile(member['login'])
    if userData:
      contrib_database[member['login']] = userData
  WriteContributorDatabase(contrib_database)

def PrintStats():
  registered = 0
  missing = 0
  noreply = 0
  contrib_database = LoadContributorDatabase()
  for login, record in contrib_database.items():
    if 'email' not in record:
      missing += 1
      continue
    if 'noreply.github.com' in record['email']:
      noreply += 1
      continue
    registered += 1
  print('%d (%d%%) fully registered' % (registered, (registered/len(contrib_database)) * 100))
  print('%d (%d%%) missing email' % (missing, (missing/len(contrib_database)) * 100))
  print('%d (%d%%) using noreply' % (noreply, (noreply/len(contrib_database)) * 100))
  print('%d total records' % len(contrib_database))

def LoadFilteredDatabase():
  contrib_database = LoadContributorDatabase()
  filter = ParseArgs().filter
  if filter == 'all':
    return contrib_database
  if filter == 'missing':
    return { key: value for key, value in contrib_database.items() if 'email' not in value }
  if filter == 'noreply':
    return { key: value for key, value in contrib_database.items() if 'email' in value and 'noreply.github' in value['email'] }
  if filter == 'complete':
    return { key: value for key, value in contrib_database.items() if 'email' in value and 'noreply.github' not in value['email'] }
  return contrib_database

def PrintUser(user):
  if 'email' in user:
    print('User: %s <%s>' % (user['login'], user['email']))
  else:
    print('User: %s' % user['login'])

def Print():
  contrib_database = LoadFilteredDatabase()
  for key, value in contrib_database.items():
    PrintUser(value)

def QuerySingleUser():
  args = ParseArgs().extra_args
  if len(args) == 0:
    print('query command expects a list of usernames to query for')
    return
  for user in args:
    userData = GenerateUserProfile(user)
    PrintUser(userData)

def main():
  args = ParseArgs()
  if args.verbose:
    print('Beginning processing - %s' % str(start_time))
  if args.action == 'register':
    RegisterContributors()
  if args.action == 'stats':
    PrintStats()
  if args.action == 'print':
    Print()
  if args.action == 'query':
    QuerySingleUser()

  if args.verbose:
    print('Exiting - %s' % str(datetime.now()))

if __name__ == '__main__':
  main()
