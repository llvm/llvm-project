%module lldbIntelFeatures

%{
#include "lldb/lldb-public.h"
#include "intel-pt/PTDecoder.h"
using namespace ptdecoder;
%}

/* Undefine GCC keyword to make Swig happy when processing glibc's stdint.h */
#define __extension__

/* Combined python typemap for all features */
%include "python-typemaps.txt"

/* Feature specific python interface files*/
%include "../intel-pt/interface/PTDecoder.i"
