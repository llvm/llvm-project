#!/usr/bin/perl

use DBI;
use CGI;

$q = new CGI;
print $q->header();
print $q->start_html(-title=>"Nightly Tester DB");

unless($q->param('pwd'))
  {
    print $q->startform();
    print $q->password_field(-name=>"pwd", -size=>20, -maxlength=>20);
    print $q->submit();
    print $q->endform();
  }
else
  {
    # database information
    $db="llvmalpha";
    $host="localhost";
    $userid="llvmdbuser";
    $passwd=$q->param('pwd');
    $connectionInfo="dbi:mysql:$db;$host";
    
    # make connection to database
    $dbh = DBI->connect($connectionInfo,$userid,$passwd) or die DBI->errstr;
    $query = "Select DISTINCT(NAME) from Tests";
    my $sth = $dbh->prepare($query) || die "Can't prepare statement: $DBI::errstr";
    my $rc = $sth->execute or die DBI->errstr;
    while (($n) = $sth->fetchrow_array)
      {
        push @names, ($n);
#        print "$n<P>";
      }
    $query = "Select DISTINCT(TEST) from Tests";
    my $sth = $dbh->prepare($query) || die "Can't prepare statement: $DBI::errstr";
    my $rc = $sth->execute or die DBI->errstr;
    while (($n) = $sth->fetchrow_array)
      {
        push @tests, ($n);
#        print "$n\n";
      }

#    print join "<BR>", @names;

    print $q->startform();
    print $q->scrolling_list(-name=>"test", -values=>\@tests, -multiple=>'true');
    print "<P>";
    print $q->scrolling_list(-name=>"name", -values=>\@names, -multiple=>'true');
    print "<P>";
    print $q->submit();
    print $q->hidden("pwd", $q->param('pwd'));
    print $q->endform();

    # disconnect from database
    $dbh->disconnect;

    #now generate the urls to the chart
    if ($q->param('test') && $q->param('name'))
      {
        my @names = $q->param('name');
        my @tests = $q->param('test');
        print "<P>";
        print join "<BR>", @names;
        print "<P>";
        print join "<BR>", @tests;
        print "<P>";
        $str = "pwd=" . $q->param('pwd');
        $count = 0;
        foreach $n (@names)
          {
            foreach $t (@tests)
              {
                $str = "$str&t$count=$t&n$count=$n";
                $count++;
              }
          }
        print "<img src=\"cgiplotNLT.pl?$str\">";
      }
  }

print $q->end_html();
