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
	(./isl_schedule$EXEEXT $options < $i > $test &&
	./isl_schedule_cmp$EXEEXT $ref $test && rm $test) || failed=1
done

test $failed -eq 0 || exit
