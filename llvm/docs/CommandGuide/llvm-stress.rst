llvm-stress - generate random .ll files
=======================================


SYNOPSIS
--------


**llvm-stress** [-size=filesize] [-seed=initialseed] [-o=outfile]


DESCRIPTION
-----------


The **llvm-stress** tool is used to generate random .ll files that can be used to
test different components of LLVM.


OPTIONS
-------



**-o** *filename*

 Specify the output filename.



**-size** *size*

 Specify the size of the generated .ll file.



**-seed** *seed*

 Specify the seed to be used for the randomly generated instructions.




EXIT STATUS
-----------


**llvm-stress** returns 0.
