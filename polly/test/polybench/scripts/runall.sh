#!/bin/sh

if [ $# -ne 1 ]; then
    echo "Usage: runall.sh <machine-acronym>";
    exit 1;
fi;

## Default value for the compilation line.
if [ -z "$COMPILER_COMMAND" ]; then
    COMPILER_COMMAND="gcc -O3 -fopenmp";
fi;

echo "Machine: $1";
for i in `ls`; do
    if [ -d "$i" ] && [ -f "$i/$i.c" ]; then
	echo "Testing benchmark $i";
	rm -f data/$1-$i.dat
	if [ -f "$i/compiler.opts" ]; then
	    read comp_opts < $i/compiler.opts;
	    COMPILER_F_COMMAND="$COMPILER_COMMAND $comp_opts";
	else
	    COMPILER_F_COMMAND="$COMPILER_COMMAND";
	fi;
	for j in `find $i -name "*.c"`; do
	    echo "Testing $j";
	    scripts/compile.sh "$COMPILER_F_COMMAND" "$j" "transfo" > /dev/null;
	    if [ $? -ne 0 ]; then
		echo "Problem when compiling $j";
	    else
		val=`./transfo`;
		if [ $? -ne 0 ]; then
		    echo "Problem when executing $j";
		else
		    echo "execution time: $val";
		    echo "$j $val" >> data/$1-$i.dat
		fi;
	    fi;
	done;
    fi;
done;
