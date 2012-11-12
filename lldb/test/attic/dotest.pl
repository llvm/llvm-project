#!/usr/bin/perl -w

#
# Use this script to visit each python test case under the specified directory
# and invoke unittest.main() on each test case.
#

use strict;
use FindBin;
use File::Find;
use File::Basename;
use Cwd;
use Cwd 'abs_path';

scalar(@ARGV) == 1 or die "Usage: dotest.pl testdir";

my $scriptDir = $FindBin::Bin;
my $baseDir = abs_path("$scriptDir/..");
my $pluginDir = "$baseDir/test/plugins";
my $testDir = $ARGV[0];

my $dbgPath = "$baseDir/build/Debug/LLDB.framework/Resources/Python";
my $relPath = "$baseDir/build/Release/LLDB.framework/Resources/Python";
if (-d $dbgPath) {
  $ENV{'PYTHONPATH'} = "$dbgPath:$scriptDir:$pluginDir";
} elsif (-d $relPath) {
  $ENV{'PYTHONPATH'} = "$relPath:$scriptDir:$pluginDir";
}
#print("ENV{PYTHONPATH}=$ENV{'PYTHONPATH'}\n");

# Traverse the directory to find our python test cases.
find(\&handleFind, $testDir);

sub handleFind {
  my $foundFile = $File::Find::name;
  my $dir = getcwd;
  #print("foundFile: $foundFile\n");
  
  # Test*.py is the naming pattern for our test cases.
  if ($foundFile =~ /.*\/(Test.*\.py)$/) {
    print("Running python $1 (cwd = $dir)...\n");
    system("python $1");
  }
}
