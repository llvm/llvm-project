@echo off
rem doflex.cmd prefix mode target source
rem   mode - either debug or release
rem   target - generated parser file name without extension
rem   source - input to bison

if "%1"=="debug" (set flags=-t) else (set flags=-t)

rem Test for presence of flex.
flex --help >NUL
if errorlevel 1 goto noflex

rem Run flex.
flex %flags% >%2.cpp %3
exit

:noflex
echo Flex not found.  Using pre-generated files.
copy %~pn3.cpp.cvs %2.cpp
exit
