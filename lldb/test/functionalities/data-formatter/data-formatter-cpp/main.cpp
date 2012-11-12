//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

typedef float RealNumber; // should show as char
typedef RealNumber Temperature; // should show as float
typedef RealNumber Speed; // should show as hex

typedef int Counter; // should show as int
typedef int BitField; // should show as hex

typedef BitField SignalMask; // should show as hex
typedef BitField Modifiers; // should show as hex

typedef Counter Accumulator; // should show as int

typedef int Type1; // should show as char
typedef Type1 Type2; // should show as hex
typedef Type2 Type3; // should show as char
typedef Type3 Type4; // should show as char

typedef int ChildType; // should show as int
typedef int AnotherChildType; // should show as int

struct Point {
    int x;
    int y;
    Point(int X = 3, int Y = 2) : x(X), y(Y) {}
};

typedef float ShowMyGuts;

struct i_am_cool
{
	int integer;
	ShowMyGuts floating;
	char character;
	i_am_cool(int I, ShowMyGuts F, char C) :
    integer(I), floating(F), character(C) {}
	i_am_cool() : integer(1), floating(2), character('3') {}
    
};

struct i_am_cooler
{
	i_am_cool first_cool;
	i_am_cool second_cool;
	ShowMyGuts floating;
	
	i_am_cooler(int I1, int I2, float F1, float F2, char C1, char C2) :
    first_cool(I1,F1,C1),
    second_cool(I2,F2,C2),
    floating((F1 + F2)/2) {}
};

struct IUseCharStar
{
	const char* pointer;
	IUseCharStar() : pointer("Hello world") {}
};

int main (int argc, const char * argv[])
{
    
    int iAmInt = 1;
    const float& IAmFloat = float(2.45);

    RealNumber RNILookChar = 3.14;
    Temperature TMILookFloat = 4.97;
    Speed SPILookHex = 5.55;
    
    Counter CTILookInt = 6;
    BitField BFILookHex = 7;
    SignalMask SMILookHex = 8;
    Modifiers MFILookHex = 9;
    
    Accumulator* ACILookInt = new Accumulator(10);
    
    const Type1& T1ILookChar = 11;
    Type2 T2ILookHex = 12;
    Type3 T3ILookChar = 13;
    Type4 T4ILookChar = 14;
    
    AnotherChildType AHILookInt = 15;
    
    Speed* SPPtrILookHex = new Speed(16);
    
    Point iAmSomewhere(4,6);
    
	i_am_cool *cool_pointer = (i_am_cool*)malloc(sizeof(i_am_cool)*3);
	cool_pointer[0] = i_am_cool(3,-3.141592,'E');
	cool_pointer[1] = i_am_cool(0,-3.141592,'E');
	cool_pointer[2] = i_am_cool(0,-3.141592,'E');
    
    i_am_cool cool_array[5];
    
    cool_array[3].floating = 5.25;
    cool_array[4].integer = 6;
    cool_array[2].character = 'Q';
    
    int int_array[] = {1,2,3,4,5};
    
    IUseCharStar iEncapsulateCharStar;
    
    char  strarr[32] = "Hello world!";
    char* strptr     = "Hello world!";
    
    i_am_cooler the_coolest_guy(1,2,3.14,6.28,'E','G');
        
    return 0; // Set break point at this line.
}

