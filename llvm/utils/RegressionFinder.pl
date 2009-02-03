#! /usr/bin/perl
# Script to find regressions by binary-searching a time interval in the
# CVS tree.  Written by Brian Gaeke on 2-Mar-2004.
#

require 5.6.0;  # NOTE: This script not tested with earlier versions.
use Getopt::Std;
use POSIX;
use Time::Local;
use IO::Handle;

sub usage {
    print STDERR <<END;
findRegression [-I] -w WTIME -d DTIME -t TOOLS -c SCRIPT

The -w, -d, -t, and -c options are required.
Run findRegression in the top level of an LLVM tree.
WTIME is a time when you are sure the regression does NOT exist ("Works").
DTIME is a time when you are sure the regression DOES exist ("Doesntwork").
WTIME and DTIME are both in the format: "YYYY/MM/DD HH:MM".
-I means run builds at WTIME and DTIME first to make sure.
TOOLS is a comma separated list of tools to rebuild before running SCRIPT.
SCRIPT exits 1 if the regression is present in TOOLS; 0 otherwise.
END
    exit 1;
}

sub timeAsSeconds {
    my ($timestr) = @_;

    if ( $timestr =~ /(\d\d\d\d)\/(\d\d)\/(\d\d) (\d\d):(\d\d)/ ) {
        my ( $year, $mon, $mday, $hour, $min ) = ( $1, $2, $3, $4, $5 );
        return timegm( 0, $min, $hour, $mday, $mon - 1, $year );
    }
    else {
        die "** Can't parse date + time: $timestr\n";
    }
}

sub timeAsString {
    my ($secs) = @_;
    return strftime( "%Y/%m/%d %H:%M", gmtime($secs) );
}

sub run {
    my ($cmdline) = @_;
    print LOG "** Running: $cmdline\n";
	return system($cmdline);
}

sub buildLibrariesAndTools {
    run("sh /home/vadve/gaeke/scripts/run-configure");
    run("$MAKE -C lib/Support");
    run("$MAKE -C utils");
    run("$MAKE -C lib");
    foreach my $tool (@TOOLS) { run("$MAKE -C tools/$tool"); }
}

sub contains {
    my ( $file, $regex ) = @_;
    local (*FILE);
    open( FILE, "<$file" ) or die "** can't read $file: $!\n";
    while (<FILE>) {
        if (/$regex/) {
            close FILE;
            return 1;
        }
    }
    close FILE;
    return 0;
}

sub updateSources {
    my ($time) = @_;
    my $inst = "include/llvm/Instruction.h";
    unlink($inst);
    run( "cvs update -D'" . timeAsString($time) . "'" );
    if ( !contains( $inst, 'class Instruction.*Annotable' ) ) {
        run("patch -F100 -p0 < makeInstructionAnnotable.patch");
    }
}

sub regressionPresentAt {
    my ($time) = @_;

    updateSources($time);
    buildLibrariesAndTools();
    my $rc = run($SCRIPT);
    if ($rc) {
        print LOG "** Found that regression was PRESENT at "
          . timeAsString($time) . "\n";
        return 1;
    }
    else {
        print LOG "** Found that regression was ABSENT at "
          . timeAsString($time) . "\n";
        return 0;
    }
}

sub regressionAbsentAt {
    my ($time) = @_;
    return !regressionPresentAt($time);
}

sub closeTo {
    my ( $time1, $time2 ) = @_;
    return abs( $time1 - $time2 ) < 600;    # 10 minutes seems reasonable.
}

sub halfWayPoint {
    my ( $time1, $time2 ) = @_;
    my $halfSpan = int( abs( $time1 - $time2 ) / 2 );
    if ( $time1 < $time2 ) {
        return $time1 + $halfSpan;
    }
    else {
        return $time2 + $halfSpan;
    }
}

sub checkBoundaryConditions {
    print LOG "** Checking for presence of regression at ", timeAsString($DTIME),
      "\n";
    if ( !regressionPresentAt($DTIME) ) {
        die ( "** Can't help you; $SCRIPT says regression absent at dtime: "
              . timeAsString($DTIME)
              . "\n" );
    }
    print LOG "** Checking for absence of regression at ", timeAsString($WTIME),
      "\n";
    if ( !regressionAbsentAt($WTIME) ) {
        die ( "** Can't help you; $SCRIPT says regression present at wtime: "
              . timeAsString($WTIME)
              . "\n" );
    }
}

##############################################################################

# Set up log files
open (STDERR, ">&STDOUT") || die "** Can't redirect std.err: $!\n";
autoflush STDOUT 1;
autoflush STDERR 1;
open (LOG, ">RegFinder.log") || die "** can't write RegFinder.log: $!\n";
autoflush LOG 1;
# Check command line arguments and environment variables
getopts('Iw:d:t:c:');
if ( !( $opt_w && $opt_d && $opt_t && $opt_c ) ) {
    usage;
}
$MAKE  = $ENV{'MAKE'};
$MAKE  = 'gmake' unless $MAKE;
$WTIME = timeAsSeconds($opt_w);
print LOG "** Assuming worked at ", timeAsString($WTIME), "\n";
$DTIME = timeAsSeconds($opt_d);
print LOG "** Assuming didn't work at ", timeAsString($DTIME), "\n";
$opt_t =~ s/\s*//g;
$SCRIPT = $opt_c;
die "** $SCRIPT is not executable or not found\n" unless -x $SCRIPT;
print LOG "** Checking for the regression using $SCRIPT\n";
@TOOLS = split ( /,/, $opt_t );
print LOG (
    "** Going to rebuild: ",
    ( join ", ", @TOOLS ),
    " before each $SCRIPT run\n"
);
if ($opt_I) { checkBoundaryConditions(); }
# do the dirty work:
while ( !closeTo( $DTIME, $WTIME ) ) {
    my $halfPt = halfWayPoint( $DTIME, $WTIME );
    print LOG "** Checking whether regression is present at ",
      timeAsString($halfPt), "\n";
    if ( regressionPresentAt($halfPt) ) {
        $DTIME = $halfPt;
    }
    else {
        $WTIME = $halfPt;
    }
}
# Tell them what we found
print LOG "** Narrowed it down to:\n";
print LOG "** Worked at: ",       timeAsString($WTIME), "\n";
print LOG "** Did not work at: ", timeAsString($DTIME), "\n";
close LOG;
exit 0;
