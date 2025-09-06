=================================
SYCL Front-end device compilation
=================================

.. contents::
   :local:

.. _sycl_frontend_device_compilation:
SYCL front-end device compiler parses input source, outlines device part of the
code, applies additional restrictions on the device code (e.g. no exceptions or
virtual calls), generates LLVM IR bitcode for the device code only and
"integration header" which provides information like kernel name, parameters
order and data type for the runtime library. Multiple LLVM IR bitcodes (in case
of multiple targets) are packaged into a single object by the
clang-offload-packager.

An example of front-end device compilation command is shown below:

  .. code-block:: console

    $ clang -cc1 -triple spirv64 -fsycl-is-device test.cpp -o test.bc

Front-end device compilation for SYCL offloading can be split into the following
components - Device code outlining, SYCL kernel function object lowering,
Generation of device code diagnostics, and Integration header generation. These
components are explained in the sections below.

*********************
Device code outlining
*********************
  This component is responsible for identifying and outlining "device code" in the
  single source.
  Here is a code example of a SYCL program that demonstrates compiler outlining
  work:

  .. code-block:: c++
    :linenos:
    
      int foo(int x) { return ++x; }
      int bar(int x) { throw std::exception{"CPU code only!"}; }
      // ...
      using namespace sycl;
      queue Q;
      buffer<int, 1> a{range<1>{1024}};
      Q.submit([&](handler& cgh) {
        auto A = a.get_access<access::mode::write>(cgh);
        cgh.parallel_for<init_a>(range<1>{1024}, [=](id<1> index) {
          A[index] = index[0] * 2 + foo(42);
        });
      }

  In this example, the compiler needs to compile the lambda expression passed
  to the `sycl::handler::parallel_for` method, as well as the function `foo`
  called from the lambda expression for the device.

  The compiler must also ignore the `bar` function when we compile the
  "device" part of the single source code, as it's unused inside the device
  portion of the source code (the contents of the lambda expression passed to the
  `sycl::handler::parallel_for` and any function called from this lambda
  expression).

  The current approach is to use the SYCL kernel attribute in the runtime to
  mark code passed to `sycl::handler::parallel_for` as "kernel functions".
  The runtime library can't mark foo as "device" code - this is a compiler
  job: to traverse all symbols accessible from kernel functions and add them to
  the "device part" of the code marking them with the new SYCL device attribute.

********************************************************
SYCL kernel function object (functor or lambda) lowering
********************************************************
  This component creates an OpenCL kernel function interface for SYCL kernels.
  All SYCL memory objects shared between host and device (buffers/images,
  these objects map to OpenCL buffers and images) must be accessed through special
  `accessor` classes. The "device" side implementation of these classes contains
  pointers to the device memory. As there is no way in OpenCL to pass structures
  with pointers inside as kernel arguments all memory objects shared between host
  and device must be passed to the kernel as raw pointers.

  SYCL also has a special mechanism for passing kernel arguments from host to
  the device. In OpenCL kernel arguments are set by calling `clSetKernelArg`
  function for each kernel argument, meanwhile in SYCL all the kernel arguments
  are fields of "SYCL kernel function" which can be defined as a lambda function
  or a named function object and passed as an argument to SYCL function for
  invoking kernels (such as `parallel_for` or `single_task`). For example, in the
  previous code snippet above `accessor` `A` is one such captured kernel argument.

  To facilitate the mapping of SYCL kernel data members to OpenCL
  kernel arguments and overcome OpenCL limitations we added the generation of an
  OpenCL kernel function inside the compiler. An OpenCL kernel function contains
  the body of the SYCL kernel function, receives OpenCL-like parameters and
  additionally does some manipulation to initialize SYCL kernel data members
  with these parameters. In some pseudo code the OpenCL kernel function for the
  previous code snippet above looks like this:

  .. code-block:: c++
    :linenos:

      // SYCL kernel is defined in SYCL headers:
      template <typename KernelName, typename KernelType/*, ...*/>
      __attribute__((sycl_kernel)) void sycl_kernel_function(KernelType KernelFuncObj) {
        // ...
        KernelFuncObj();
      }
      // Generated OpenCL kernel function
      __kernel KernelName(global int* a) {
        KernelType KernelFuncObj; // Actually kernel function object declaration
        // doesn't have a name in AST.
        // Let the kernel function object have one captured field - accessor A.
        // We need to init it with global pointer from arguments:
        KernelFuncObj.A.__init(a);
        // Body of the SYCL kernel from SYCL headers:
        {
          KernelFuncObj();
        }
      }

  OpenCL kernel function is generated by the compiler inside the Sema using AST
  nodes.

*************************************
Generation of device code diagnostics
*************************************
  This component enforces language restrictions on device code.

*****************************
Integration header generation
*****************************
  This component emits information required for binding host and device parts of
  the SYCL code via OpenCL API. In proposed design, we use SYCL device front-end
  compiler to produce the integration header for two reasons. First, it must be
  possible to use any host compiler to produce SYCL heterogeneous applications.
  Second, even if the same clang compiler is used for the host compilation,
  information provided in the integration header is used (included) by the SYCL
  runtime implementation, so the header must be available before the host
  compilation starts.