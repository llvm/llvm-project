#  Copyright John Maddock 2008.
#  Use, modification and distribution are subject to the
#  Boost Software License, Version 1.0. (See accompanying file
#  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#
# Generates PNG files from the SVG masters.
#
# Paths to tools come first, change these to match your system:
#
math2svg='m:\download\open\SVGMath-0.3.1\math2svg.py'
python='/cygdrive/c/program files/Python27/python.exe'
inkscape='/cygdrive/c/Program Files (x86)/Inkscape/inkscape.exe'
# Image DPI:
dpi=96

for svgfile in $*; do
	pngfile=$(basename $svgfile .svg).png
	echo Generating $pngfile
	"$inkscape" -d $dpi -e $(cygpath -a -w $pngfile) $(cygpath -a -w $svgfile)
done








