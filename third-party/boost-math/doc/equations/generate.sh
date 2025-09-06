#  Copyright John Maddock 2008.
#  Use, modification and distribution are subject to the
#  Boost Software License, Version 1.0. (See accompanying file
#  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#
# Generates SVG and PNG files from the MML masters.
#
# Paths to tools come first, change these to match your system:
#
math2svg='d:\download\open\SVGMath-0.3.3\math2svg.py'
python='/cygdrive/c/Python27/python.exe'
inkscape='/cygdrive/c/Program Files/Inkscape/inkscape.exe'
# Image DPI:
dpi=120

for mmlfile in $*; do
	svgfile=$(basename $mmlfile .mml).svg
	pngfile=$(basename $svgfile .svg).png
	tempfile=temp.mml
	# strip html wrappers put in by MathCast:
	cat $mmlfile | tr -d "\r\n" | sed -e 's/.*\(<math[^>]*>.*<\/math>\).*/\1/' -e 's/<semantics>//g' -e 's/<\/semantics>//g' > $tempfile
	
	echo Generating $svgfile
	"$python" $math2svg $tempfile > $svgfile
	echo Generating $pngfile
	"$inkscape" -d $dpi -e $(cygpath -a -w $pngfile) $(cygpath -a -w $svgfile)
	rm $tempfile
done




