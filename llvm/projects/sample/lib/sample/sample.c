/*
 * File: sample.c
 *
 * Description:
 *  This is a sample source file for a library.  It helps to demonstrate
 *  how to setup a project that uses the LLVM build system, header files,
 *  and libraries.
 */

#include <stdio.h>
#include <stdlib.h>

/* LLVM Header File
#include "llvm/Support/DataTypes.h"
*/

/* Header file global to this project */
#include "sample.h"

int
compute_sample (int a)
{
  return a;
}

