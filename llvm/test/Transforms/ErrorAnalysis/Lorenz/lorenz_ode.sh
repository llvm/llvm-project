#! /bin/bash
#
gcc -c -Wall lorenz_ode.c
if [ $? -ne 0 ]; then
  echo "Compile error."
  exit
fi
#
gcc lorenz_ode.o
if [ $? -ne 0 ]; then
  echo "Load error."
  exit
fi
rm lorenz_ode.o
#
echo "Normal end of execution."
