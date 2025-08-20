# Building orc-rt

## Getting Started

The basic steps needed to build orc-rt are:

* Checkout llvm-project:

   * ``cd where-you-want-llvm-to-live``
   * ``git clone https://github.com/llvm/llvm-project.git``

* Configure and build orc-rt:

   CMake is the only supported configuration system.

   Clang is the preferred compiler when building and using orc-rt.

   * ``cd where you want to build llvm``
   * ``mkdir build``
   * ``cd build``
   * ``cmake -G <generator> -DLLVM_ENABLE_RUNTIMES=orc-rt [options] <llvm-monorepo>/runtimes``

   For more information about configuring orc-rt see :ref:`CMake Options`.

   * ``make orc-rt`` --- will build orc-rt.
   * ``make check-orc-rt`` --- will run the test suite.

   Shared and static libraries for orc-rt should now be present in
   llvm/build/lib.

* **Optional**: Install orc-rt

   Remember to use the CMake option ``CMAKE_INSTALL_PREFIX`` to select a safe
   place to install orc-rt.

   * ``make install-orc-rt`` --- Will install the libraries and the headers

## CMake Options

Here are some of the CMake variables that are used often, along with a
brief explanation and LLVM-specific notes. For full documentation, check the
CMake docs or execute ``cmake --help-variable VARIABLE_NAME``.

**CMAKE_BUILD_TYPE**:STRING
  Sets the build type for ``make`` based generators. Possible values are
  Release, Debug, RelWithDebInfo and MinSizeRel. On systems like Visual Studio
  the user sets the build type with the IDE settings.

**CMAKE_INSTALL_PREFIX**:PATH
  Path where LLVM will be installed if "make install" is invoked or the
  "INSTALL" target is built.

**CMAKE_CXX_COMPILER**:STRING
  The C++ compiler to use when building and testing orc-rt.

## orc-rt specific options

* option:: ORC_RT_ENABLE_ASSERTIONS:BOOL

  **Default**: ``ON``

  Toggle assertions independent of the build mode.

* option:: ORC_RT_ENABLE_PEDANTIC:BOOL

  **Default**: ``ON``

  Compile with -Wpedantic.

* option:: ORC_RT_ENABLE_WERROR:BOOL

  **Default**: ``ON``

  Compile with -Werror
