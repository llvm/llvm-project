#!/usr/bin/perl
#take the output of parseNLT.pl and load it into a database
# use like: cat file |perl parseNLT.pl |perl importNLT.pl password

use DBI;

# database information
$db="llvmalpha";
$host="localhost";
$userid="llvmdbuser";
$passwd=shift @ARGV;
$connectionInfo="dbi:mysql:$db;$host";

# make connection to database
$dbh = DBI->connect($connectionInfo,$userid,$passwd) or die DBI->errstr;
my $sth = $dbh->prepare( q{
      INSERT INTO Tests (NAME, RUN, TEST, VALUE) VALUES (?, STR_TO_DATE(?, '\%d \%M \%Y'), ?, ?)
  }) || die "Can't prepare statement: $DBI::errstr";;

while($d = <>)
{
  chomp $d;
  if (18 == scalar split " ", $d)
    {
      ($day, $mon, $year, $prog, $gccas, $bc, $llccompile, $llcbetacompile, $jitcompile,
       $mc, $gcc, $cbe, $llc, $llcbeta, $jit, $foo1, $foo2, $foo3) = split " ", $d;
      if ($gccas =~ /\d+/)
        {
          $dbh->do("INSERT INTO Tests (NAME, RUN, TEST, VALUE) VALUES
                ('$prog', STR_TO_DATE('$day $mon $year', '\%d \%M \%Y'), 'gccas', $gccas)") || die DBI->errstr;
        }
      if ($bc =~ /\d/)
        {
          $dbh->do("INSERT INTO Tests (NAME, RUN, TEST, VALUE) VALUES
                ('$prog', STR_TO_DATE('$day $mon $year', '\%d \%M \%Y'), 'bytecode', $bc)") || die DBI->errstr;
        }
      if ($llccompile =~ /\d/)
        {
          $dbh->do("INSERT INTO Tests (NAME, RUN, TEST, VALUE) VALUES
                ('$prog', STR_TO_DATE('$day $mon $year', '\%d \%M \%Y'), 'llc-compile', $llccompile)") || die DBI->errstr;
        }
      if ($llcbetacompile =~ /\d/)
        {
          $dbh->do("INSERT INTO Tests (NAME, RUN, TEST, VALUE) VALUES
                ('$prog', STR_TO_DATE('$day $mon $year', '\%d \%M \%Y'), 'llc-beta-compile', $llcbetacompile)") || die DBI->errstr;
        }
      if ($jitcompile =~ /\d/)
        {
          $dbh->do("INSERT INTO Tests (NAME, RUN, TEST, VALUE) VALUES
                ('$prog', STR_TO_DATE('$day $mon $year', '\%d \%M \%Y'), 'jit-compile', $jitcompile)") || die DBI->errstr;
        }
      if ($mc =~ /\d/)
        {
          $dbh->do("INSERT INTO Tests (NAME, RUN, TEST, VALUE) VALUES
                ('$prog', STR_TO_DATE('$day $mon $year', '\%d \%M \%Y'), 'machine-code', $mc)") || die DBI->errstr;
        }
      if ($gcc =~ /\d/)
        {
          $dbh->do("INSERT INTO Tests (NAME, RUN, TEST, VALUE) VALUES
                ('$prog', STR_TO_DATE('$day $mon $year', '\%d \%M \%Y'), 'gcc', $gcc)") || die DBI->errstr;
        }
      if ($llc =~ /\d/)
        {
          $dbh->do("INSERT INTO Tests (NAME, RUN, TEST, VALUE) VALUES
                ('$prog', STR_TO_DATE('$day $mon $year', '\%d \%M \%Y'), 'llc', $llc)") || die DBI->errstr;
        }
      if ($llcbeta =~ /\d/)
        {
          $dbh->do("INSERT INTO Tests (NAME, RUN, TEST, VALUE) VALUES
                ('$prog', STR_TO_DATE('$day $mon $year', '\%d \%M \%Y'), 'llc-beta', $llcbeta)") || die DBI->errstr;
        }
      if ($jit =~ /\d/)
        {
          $dbh->do("INSERT INTO Tests (NAME, RUN, TEST, VALUE) VALUES
                ('$prog', STR_TO_DATE('$day $mon $year', '\%d \%M \%Y'), 'jit', $jit)") || die DBI->errstr;
        }
      print ".";
    }
  else
    {
      print "\nNO: $d\n";
    }
}
print "\n";
# disconnect from database
$dbh->disconnect;
