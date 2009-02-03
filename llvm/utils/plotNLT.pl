#!/usr/bin/perl
#takes a test and a program from a dp and produces a gnuplot script
#use like perl plotNLT.pl password Programs/MultiSource/Benchmarks/ASCI_Purple/SMG2000/smg2000 llc

use DBI;

# database information
$db="llvmalpha";
$host="localhost";
$userid="llvmdbuser";
$passwd=shift @ARGV;
$connectionInfo="dbi:mysql:$db;$host";

# make connection to database
$dbh = DBI->connect($connectionInfo,$userid,$passwd) or die DBI->errstr;


$count = @ARGV / 2;

print "set xdata time\n";
print 'set timefmt "%Y-%m-%d"';
print "\nplot";
for ($iter = 0; $iter < $count; $iter++) {
  if ($iter)
    { print ","; }
  print " '-' using 1:2 with lines";
}

print "\n";

for ($iter = 0; $iter < $count; $iter++) {

  $prog = shift @ARGV;
  $test = shift @ARGV;

  $query = "Select RUN, VALUE from Tests where TEST = '$test' AND NAME = '$prog' ORDER BY RUN";
  #print "\n$query\n";
  
  my $sth = $dbh->prepare( $query) || die "Can't prepare statement: $DBI::errstr";;
  
  my $rc = $sth->execute or die DBI->errstr;
  
  while(($da,$v) = $sth->fetchrow_array)
    {
      print "$da $v\n";
    }
  
  print "e\n";
}


# disconnect from database
$dbh->disconnect;
