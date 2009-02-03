@echo off
rem dobison.cmd prefix mode target source
rem   prefix - passed to bison as -p<prefix>
rem   mode - either debug or release
rem   target - generated parser file name without extension
rem   source - input to bison
rem   headercopydir - directory to receive a copy of the header

if "%2"=="debug" (set flags=-tvdo) else (set flags=-vdo)

rem Test for presence of bison.
bison --help >NUL
if errorlevel 1 goto nobison

rem Run bison.
echo bison -p%1 %flags%%3.cpp %4
echo move %3.hpp %3.h
bison -p%1 %flags%%3.cpp %4 && move %3.hpp %3.h
echo copy %3.h %5
copy %3.h %5
exit

:nobison
echo Bison not found.  Using pre-generated files.
copy %~pn4.cpp.cvs .\%3.cpp
copy %~pn4.cpp.cvs %5\%3.cpp
copy %~pn4.h.cvs .\%3.h
copy %~pn4.h.cvs %5\%3.h
exit
