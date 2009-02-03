#this script is intended to help recover the running graphs when
#the nightly tester decides to eat them.

#zgrep -E "(=========)|(TEST-RESULT-llc-time)" *-Olden-tests.txt* |perl this > file
#zgrep -E "(=========)|(TEST-RESULT-compile.*bc)" *-Olden-tests.tx* |perl this >file

while (<>) {
  if (/(\d*-\d*-\d*)-.*=========.*\/(.*)\' Program/) {
#    print "$1 $2\n";
    $curP = $2;
    $curD = $1;
    $dates{$1} = 1;
  } elsif (/(\d*-\d*-\d*)-.*TEST-RESULT-.*: program (\d*\.\d*)/) {
#    print "$1 $2\n";
    if ($curD eq $1) {
      $$data{$curD}{$curP} = $2;
    }
  } elsif (/(\d*-\d*-\d*)-.*TEST-RESULT-.*: (\d*)/) {
#    print "$1 $2\n";
    if ($curD eq $1) {
      $$data{$curD}{$curP} = $2;
    }
  }
}
@progs = ("bh", "em3d", "mst", "power", "tsp", "bisort", "health", "perimeter", "treeadd", "voronoi");

foreach $date (sort keys %dates) {
  print "$date: ";
  foreach $prog (@progs) {
    if ($$data{$date}{$prog}) {
      print " $$data{$date}{$prog}";
    } else {
      print " 0";
    }
  }
  print "\n";
}
