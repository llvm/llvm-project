#!/usr/bin/perl
# a first attempt to parse the nightly tester pages into something
# one can reason about, namely import into a database
# USE: perl parseNLT.pl <2005-03-31.html
# for example

while(<>)
  {
    if (/LLVM Test Results for (\w+) (\d+), (\d+)</)
      {
        $mon = $1;
        $day = $2;
        $year = $3;
      }
    if (/<td>([^<]+)<\/td>/)
      {
        if ($prefix)
          { $output .= "$1 "; $count++; }
      }
    if (/<tr/)
      {
        if ($output and $count > 3)
          { print "\n$day $mon $year $prefix/$output"; }
	$output = "";
	$count = 0;
      }
    if (/<h2>(Programs.+)<\/h2>/)
      {
        $prefix = $1;
      }
  }

if ($output)
  { print "\n$day $mon $year $prefix/$output"; $output = ""; }
