#!/bin/sh

EXEEXT=@EXEEXT@
GREP=@GREP@
SED=@SED@
srcdir=@srcdir@

failed=0

for i in $srcdir/test_inputs/schedule/*.sc; do
	echo $i;
	base=`basename $i .sc`
	test=test-$base.st
	dir=`dirname $i`
	ref=$dir/$base.st
	options=`$GREP 'OPTIONS:' $i | $SED 's/.*://'`
	for o in --schedule-whole-component --no-schedule-whole-component; do
		./isl_schedule$EXEEXT $o $options < $i > $test &&
		    ./isl_schedule_cmp$EXEEXT $ref $test && rm $test
		if [ $? -ne 0 ]; then
			echo $o $options
			failed=1
		fi
	done
done

test $failed -eq 0 || exit
