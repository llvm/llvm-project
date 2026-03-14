#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

struct SomeData
{
    int x;
};

struct SomeOtherData
{
    char strarr[32];
    char *strptr;
    int intarr[5];
    float flarr[7];
    
    SomeOtherData()
    {
        strcpy(strarr,"Nested Hello world!");
        strptr = new char[128];
        strcpy(strptr,"Nested Hello world!");
        intarr[0] = 9;
        intarr[1] = 8;
        intarr[2] = 7;
        intarr[3] = 6;
        intarr[4] = 5;
        
        flarr[0] = 25.5;
        flarr[1] = 25.25;
        flarr[2] = 25.125;
        flarr[3] = 26.75;
        flarr[4] = 27.375;
        flarr[5] = 27.5;
        flarr[6] = 26.125;
    }
};

int main (int argc, const char * argv[])
{
    char strarr[32] = "Hello world!";
    char *strptr = NULL;
    strptr = "Hello world!";
    int intarr[5] = {1,1,2,3,5};
    float flarr[7] = {78.5,77.25,78.0,76.125,76.75,76.875,77.0};
    
    SomeData data;
    
    SomeOtherData other;
    
    float* flptr = flarr;
    int* intptr = intarr;
    
    return 0; // Set break point at this line.
    
}
