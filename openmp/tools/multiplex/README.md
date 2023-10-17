# OMPT-Multiplexing
The OMPT-Multiplexing header file allows a tool to load a second tool to 
overcome the restriction of the OpenMP to only load one tool at a time. 
The header file can also be used to load more than two tools using a cascade 
of tools that include the header file. OMPT-Multiplexing takes care of the 
multiplexing of OMPT callbacks, data pointers and runtime entry functions.

Examples can be found under ./tests

## Prerequisits
- LLVM/OpenMP runtime with OMPT (https://github.com/OpenMPToolsInterface/LLVM-openmp)
- LLVM-lit

### Getting LLVM-lit
Either build llvm and find lit+FileCheck in build directory of llvm or install using `pip`:
```
 $ pip install --upgrade --user pip
 $ export PATH=$HOME/.local/bin:$PATH
 $ export PYTHONPATH=$HOME/.local/lib/python3.*/site-packages/
 $ pip install --user lit
```

## How to test
```
 $ make check-ompt-multiplex
```

## How to compile and use your OpenMP tools
Code of first tool must include the following with the convention, that the environment variable containing the path to the client tool is the tool name with the suffix "_TOOL_LIBRARIES":
```
#define CLIENT_TOOL_LIBRARIES_VAR "EXAMPLE_TOOL_LIBRARIES"
#define CLIENT_TOOL_VERBOSE_INIT_VAR "EXAMPLE_TOOL_VERBOSE_INIT"
#include <ompt-multiplex.h>
```
Alternatively, the name of the tool can be set as a prefix for both variables:
```
#define OMPT_MULTIPLEX_TOOL_NAME "EXAMPLE"
#include <ompt-multiplex.h>
```
This define will have the same effect as to two defines above.

Note that functions and variables with prefix "ompt_multiplex" are reserved by the tool


To use both tools execute the following:
```
 $ clang -fopenmp -o program.exe
 $ OMP_TOOL_LIBRARIES=/path/to/first/tool.so EXAMPLE_TOOL_LBRARIES=/path/to/second/tool.so ./program.exe
```
Note that EXAMPLE_TOOL_LIBRARIES may also contain a list of paths to tools which will be tried to load in order (similar to lists in OMP_TOOL_LIBRARIES).

## Advanced usage
To reduce the amount of memory allocations, the user can define macros before including the ompt-multiplex.h file, that specify custom data access handlers:

```
#define OMPT_MULTIPLEX_CUSTOM_GET_CLIENT_THREAD_DATA get_client_thread_data
#define OMPT_MULTIPLEX_CUSTOM_GET_CLIENT_PARALLEL_DATA get_client_parallel_data
#define OMPT_MULTIPLEX_CUSTOM_GET_CLIENT_TASK_DATA get_client_task_data
```

This will reverse the calling order of the current tool and its client for clean-up events. In order to avoid this, one can specify a custom delete handler as well:

```
#define OMPT_MULTIPLEX_CUSTOM_DELETE_THREAD_DATA delete_thread_data
#define OMPT_MULTIPLEX_CUSTOM_DELETE_PARALLEL_DATA delete_parallel_data
#define OMPT_MULTIPLEX_CUSTOM_DELETE_TASK_DATA delete_task_data
```

