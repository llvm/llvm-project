# llvm-config - Print LLVM compilation options

```{program} llvm-config
```

## SYNOPSIS

**llvm-config** *option* \[*components*...\]

## DESCRIPTION

**llvm-config** makes it easier to build applications that use LLVM. It can
print the compiler flags, linker flags and object libraries needed to link
against LLVM.

## EXAMPLES

To link against the JIT:

```sh
g++ `llvm-config --cxxflags` -o HowToUseJIT.o -c HowToUseJIT.cpp
g++ `llvm-config --ldflags` -o HowToUseJIT HowToUseJIT.o \
    `llvm-config --libs engine bcreader scalaropts`
```

## OPTIONS

:::{option} --assertion-mode
Print the assertion mode used when LLVM was built (ON or OFF).
:::

:::{option} --bindir
Print the installation directory for LLVM binaries.
:::

:::{option} --build-mode
Print the build mode used when LLVM was built (e.g. Debug or Release).
:::

:::{option} --build-system
Print the build system used to build LLVM (e.g. `cmake` or `gn`).
:::

:::{option} --cflags
Print the C compiler flags needed to use LLVM headers.
:::

:::{option} --cmakedir
Print the installation directory for LLVM CMake modules.
:::

:::{option} --components
Print all valid component names.
:::

:::{option} --cppflags
Print the C preprocessor flags needed to use LLVM headers.
:::

:::{option} --cxxflags
Print the C++ compiler flags needed to use LLVM headers.
:::

:::{option} --has-rtti
Print whether or not LLVM was built with rtti (YES or NO).
:::

:::{option} --help
Print a summary of **llvm-config** arguments.
:::

:::{option} --host-target
Print the target triple used to configure LLVM.
:::

:::{option} --ignore-libllvm
Ignore libLLVM and link component libraries instead.
:::

:::{option} --includedir
Print the installation directory for LLVM headers.
:::

:::{option} --ldflags
Print the flags needed to link against LLVM libraries.
:::

:::{option} --libdir
Print the installation directory for LLVM libraries.
:::

:::{option} --libfiles
Similar to **--libs**, but print the full path to each library file. This is
useful when creating makefile dependencies, to ensure that a tool is relinked if
any library it uses changes.
:::

:::{option} --libnames
Similar to **--libs**, but prints the bare filenames of the libraries
without **-l** or pathnames. Useful for linking against a not-yet-installed
copy of LLVM.
:::

:::{option} --libs
Print all the libraries needed to link against the specified LLVM
*components*, including any dependencies.
:::

:::{option} --link-shared
Link the components as shared libraries.
:::

:::{option} --link-static
Link the component libraries statically.
:::

:::{option} --obj-root
Print the object root used to build LLVM.
:::

:::{option} --prefix
Print the installation prefix for LLVM.
:::

:::{option} --quote-paths
Quote and escape paths when needed, most notably when a quote, space, backslash
or dollar sign characters are present in the path.
:::

:::{option} --shared-mode
Print how the provided components can be collectively linked (`shared` or `static`).
:::

:::{option} --system-libs
Print all the system libraries needed to link against the specified LLVM
*components*, including any dependencies.
:::

:::{option} --targets-built
Print the component names for all targets supported by this copy of LLVM.
:::

:::{option} --version
Print the version number of LLVM.
:::

## COMPONENTS

To print a list of all available components, run **llvm-config
--components**. In most cases, components correspond directly to LLVM
libraries. Useful "virtual" components include:

**all**
: Includes all LLVM libraries. The default if no components are specified.

**backend**
: Includes either a native backend or the C backend.

**engine**
: Includes either a native JIT or the bitcode interpreter.

## EXIT STATUS

If **llvm-config** succeeds, it will exit with 0. Otherwise, if an error
occurs, it will exit with a non-zero value.

