#!/usr/bin/perl
#takes a test and a program from a dp and produces a gnuplot script
#use like perl plotNLT.pl password Programs/MultiSource/Benchmarks/ASCI_Purple/SMG2000/smg2000 llc

use CGI;
use DBI;
my $q = new CGI;

# database information
$db="llvmalpha";
$host="localhost";
$userid="llvmdbuser";
$passwd=$q->param('pwd');
$connectionInfo="dbi:mysql:$db;$host";

# make connection to database
$dbh = DBI->connect($connectionInfo,$userid,$passwd) or die DBI->errstr;


$count = 0;
while ($q->param('n' . $count))
  {
    $count++;
  }

$| = 1;
print "Content-type: image/png", "\n\n";

open CMDSTREAM, "|gnuplot";
#open CMDSTREAM, "|echo";

print CMDSTREAM "set terminal png\n";
print CMDSTREAM "set output\n";
print CMDSTREAM "set xdata time\n";
print CMDSTREAM 'set timefmt "%Y-%m-%d"';
print CMDSTREAM "\nplot";
for ($iter = 0; $iter < $count; $iter++) {
  if ($iter)
    { print CMDSTREAM ","; }
  print CMDSTREAM " '-' using 1:2 title \"" . $q->param('t' . $iter) . "," . $q->param('n' . $iter) . "\"with lines";
}

print CMDSTREAM "\n";

for ($iter = 0; $iter < $count; $iter++) {

  $prog = $q->param('n' . $iter);
  $test = $q->param('t' . $iter);

  $query = "Select RUN, VALUE from Tests where TEST = '$test' AND NAME = '$prog' ORDER BY RUN";
  #print "\n$query\n";
  
  my $sth = $dbh->prepare( $query) || die "Can't prepare statement: $DBI::errstr";;
  
  my $rc = $sth->execute or die DBI->errstr;
  
  while(($da,$v) = $sth->fetchrow_array)
    {
      print CMDSTREAM "$da $v\n";
    }
  
  print CMDSTREAM "e\n";
}
print CMDSTREAM "exit\n";
close CMDSTREAM;

# disconnect from database
$dbh->disconnect;
