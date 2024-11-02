Instructions to use OpenMP specific debugging support for debugging C/C++ OpenMP programs through the gdb plugin are as follows:
===============================================================================================================================

    Include libompd.so directory to LD_LIBRARY_PATH
        $ export LD_LIBRARY_PATH=<installed_dir/lib/> or <build dir/libompd/src/> :$LD_LIBRARY_PATH

    Set OMP_DEBUG to enabled
        $ export OMP_DEBUG=enabled

    Compile the program to be debugged with '-g' and '-fopenmp' options as shown for a sample C source file xyz.c
        $ clang -g -fopenmp xyz.c -o xyz.out

    NOTE:
        The program to be debugged needs to have a dynamic link dependency on 'libomp.so' for OpenMP-specific debugging to work correctly.
        The user can check this using ldd on the generated binary i.e. xyz.out

    Debug the binary xyz.out by invoking gdb with the plugin as shown below. Please note that plugin '<..>/ompd/__init__.py' should be used

        $ gdb -x <build_dir/libompd/gdb-plugin/python-module/ompd/__init__.py> or <installed_dir/share/gdb/python/ompd/__init__.py> ./xyz.out

        - The gdb command 'help ompd' lists the subcommands available for OpenMP-specific debugging.
        - The command 'ompd init' needs to be run first to load the libompd.so available in the $LD_LIBRARY_PATH environment variable, and to initialize the OMPD library.
        - The 'ompd init' command starts the program run, and the program stops at a temporary breakpoint at the OpenMP internal location ompd_dll_locations_valid().
        - The user can 'continue' from the temporary breakpoint for further debugging.
        - The user may place breakpoints at the OpenMP internal locations 'ompd_bp_thread_begin' and 'ompd_bp_thread_end' to catch the OpenMP thread begin and thread end events.
        - Similarly, 'ompd_bp_task_begin' and 'ompd_bp_task_end' breakpoints may be used to catch the OpenMP task begin and task end events; 'ompd_bp_parallel_begin' and 'ompd_bp_parallel_end' to catch OpenMP parallel begin and parallel end events.

    List of OMPD subcommands that can be used in GDB:
        - ompd init     -- Finds and initializes the OMPD library; looks for the OMPD library libompd.so under $LD_LIBRARY_PATH, and if not found, under the directory in which the OMP library libomp.so is installed.
        - ompd icvs     -- Displays the values of OpenMP Internal Control Variables.
        - ompd parallel -- Displays the details of the current and enclosing parallel regions.
        - ompd threads  -- Provides information on threads of the current context.
        - ompd bt [off | on | on continued]       -- Sets the filtering mode for "bt" output on or off, or to trace worker threads back to master threads. When ‘ompd bt on’ is used, the subsequent ‘bt’ command filters out the OpenMP runtime frames to a large extent, displaying only the user application frames. When ‘ompd bt on continued’ is used, the subsequent ‘bt’ command shows the user application frames for the current thread, and continues to trace the thread parents, up to the master thread.
        - ompd step     -- Executes "step" command into user application frames, skipping OpenMP runtime frames as much as possible.


NOTES:
      (1) Debugging code that runs on an offloading device is not supported yet.
      (2) The OMPD plugin requires an environment with Python version 3.5 or above. The gdb that is used with the OMPD plugin also needs to be based on Python version 3.5 or above.

