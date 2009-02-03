#!/usr/bin/perl -w
#
# Program:  userloc.pl
#
# Synopsis: This program uses "cvs annotate" to get a summary of how many lines
#           of code the various developres are responsible for. It takes one
#           argument, the directory to process. If the argument is not specified
#           then the cwd is used. The directory must be an LLVM tree checked out
#           from cvs. 
#
# Syntax:   userloc.pl [-tag=tag|-html... <directory>...
#
# Options:
#           -tag=tag
#               Use "tag" to select the revision (as per cvs -r option)
#           -filedetails
#               Report details about lines of code in each file for each user
#           -html
#               Generate HTML output instead of text output
#           -topdir
#               Specify where the top llvm source directory is. Otherwise the
#               llvm-config tool is used to find it.
# Directories:
#   The directories passed after the options should be relative paths to
#   directories of interest from the top of the llvm source tree, e.g. "lib"
#   or "include", etc.

die "Usage userloc.pl [-tag=tag|-html] <directories>..." 
  if ($#ARGV < 0);

my $tag = "";
my $html = 0;
my $debug = 0;
my $filedetails = "";
my $srcroot = "";
while ( defined($ARGV[0]) && substr($ARGV[0],0,1) eq '-' )
{
  if ($ARGV[0] =~ /-tag=.*/) {
    $tag = $ARGV[0];
    $tag =~ s#-tag=(.*)#$1#;
  } elsif ($ARGV[0] =~ /-filedetails/) {
    $filedetails = 1;
  } elsif ($ARGV[0] eq "-html") {
    $html = 1;
  } elsif ($ARGV[0] eq "-debug") {
    $debug = 1;
  } elsif ($ARGV[0] eq "-topdir") {
    shift; $srcroot = $ARGV[0]; shift;
  } else {
    die "Invalid option: $ARGV[0]";
  }
  shift;
}

if (length($srcroot) == 0) {
  chomp($srcroot = `llvm-config --src-root`);
}
if (! -d "$srcroot") {
  die "Invalid source root: $srcroot\n";
}
chdir($srcroot);
my $llvmdo = "$srcroot/utils/llvmdo -topdir '$srcroot'";
my %Stats;
my %FileStats;

my $annotate = "cvs -z6 annotate -lf ";
if (length($tag) > 0)
{
  $annotate = $annotate . " -r" . $tag;
}

sub GetCVSFiles
{
  my $d = $_[0];
  my $files ="";
  open FILELIST, 
    "$llvmdo -dirs \"$d\" -code-only echo |" || die "Can't get list of files with llvmdo";
  while ( defined($line = <FILELIST>) ) {
    chomp($file = $line);
    print "File: $file\n" if ($debug);
    $files = "$files $file";
  }
  return $files;
}

sub ScanDir
{
  my $Dir = $_[0];
  my $files = GetCVSFiles($Dir);

  open (DATA,"$annotate $files 2>&1 |")
    || die "Can't read cvs annotation data";

  my $curfile = "";
  while ( defined($line = <DATA>) )
  {
    chomp($line);
    if ($line =~ '^Annotations for.*') {
      $curfile = $line;
      $curfile =~ s#^Annotations for ([[:print:]]*)#$1#;
      print "Scanning: $curfile\n" if ($debug);
    } elsif ($line =~ /^[0-9.]*[ \t]*\([^)]*\):/) {
      $uname = $line;
      $uname =~ s#^[0-9.]*[ \t]*\(([a-zA-Z0-9_.-]*) [^)]*\):.*#$1#;
      $Stats{$uname}++;
      if ($filedetails) {
        $FileStats{$uname} = {} unless exists $FileStats{$uname};
        ${$FileStats{$uname}}{$curfile}++;
      }
    }
  }
  close DATA;
}

sub printStats
{
  my $dir = $_[0];
  my $hash = $_[1];
  my $user;
  my $total = 0;

  foreach $user (keys %Stats) { $total += $Stats{$user}; }

  if ($html) { 
    print "<p>Total Source Lines: $total<br/></p>\n";
    print "<table>";
    print " <tr><th style=\"text-align:right\">LOC</th>\n";
    print " <th style=\"text-align:right\">\%LOC</th>\n";
    print " <th style=\"text-align:left\">User</th>\n";
    print "</tr>\n";
  }

  foreach $user ( sort keys %Stats )
  {
    my $v = $Stats{$user};
    if (defined($v))
    {
      if ($html) {
        printf "<tr><td style=\"text-align:right\">%d</td><td style=\"text-align:right\">(%4.1f%%)</td><td style=\"text-align:left\">", $v, (100.0/$total)*$v;
        if ($filedetails) {
          print "<a href=\"#$user\">$user</a></td></tr>";
        } else {
          print $user,"</td></tr>";
        }
      } else {
        printf "%8d  (%4.1f%%)  %s\n", $v, (100.0/$total)*$v, $user;
      }
    }
  }
  print "</table>\n" if ($html);

  if ($filedetails) {
    foreach $user (sort keys %FileStats) {
      my $total = 0;
      foreach $file (sort keys %{$FileStats{$user}}) { 
        $total += ${$FileStats{$user}}{$file}
      }
      if ($html) {
        print "<table><tr><th style=\"text-align:left\" colspan=\"3\"><a name=\"$user\">$user</a></th></tr>\n";
      } else {
        print $user,":\n";
      }
      foreach $file (sort keys %{$FileStats{$user}}) {
        my $v = ${$FileStats{$user}}{$file};
        if ($html) { 
          printf "<tr><td style=\"text-align:right\">&nbsp;&nbsp;%d</td><td
          style=\"text-align:right\">&nbsp;%4.1f%%</td><td
          style=\"text-align:left\">%s</td></tr>",$v, (100.0/$total)*$v,$file;
        } else {
          printf "%8d  (%4.1f%%)  %s\n", $v, (100.0/$total)*$v, $file;
        }
      }
      if ($html) { print "</table>\n"; }
    }
  }
}


if ($html)
{
print "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.01//EN\" \"http://www.w3.org/TR/html4/strict.dtd\">\n";
print "<html>\n<head>\n";
print "  <title>LLVM LOC Based On CVS Annotation</title>\n";
print "  <link rel=\"stylesheet\" href=\"llvm.css\" type=\"text/css\"/>\n";
print "</head>\n";
print "<body><div class=\"doc_title\">LLVM LOC Based On CVS Annotation</div>\n";
print "<p>This document shows the total lines of code per user in each\n";
print "LLVM directory. Lines of code are attributed by the user that last\n";
print "committed the line. This does not necessarily reflect authorship.</p>\n";
}

my @DIRS;
if ($#ARGV > 0) {
  @DIRS = @ARGV;
} else {
  push @DIRS, 'include';
  push @DIRS, 'lib';
  push @DIRS, 'tools';
  push @DIRS, 'runtime';
  push @DIRS, 'docs';
  push @DIRS, 'test';
  push @DIRS, 'utils';
  push @DIRS, 'examples';
  push @DIRS, 'projects/Stacker';
  push @DIRS, 'projects/sample';
  push @DIRS, 'autoconf';
}
  
for $Index ( 0 .. $#DIRS) { 
  print "Scanning Dir: $DIRS[$Index]\n" if ($debug);
  ScanDir($DIRS[$Index]); 
}

printStats;

print "</body></html>\n" if ($html) ;
