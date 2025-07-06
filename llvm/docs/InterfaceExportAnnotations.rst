LLVM Interface Export Annotations
=================================
Symbols that are part of LLVM's public interface must be explicitly annotated
to support shared library builds with hidden default symbol visibility. This
document provides background and guidelines for annotating the codebase.

LLVM Shared Library
-------------------
LLVM builds as a static library by default, but it can also be built as a shared
library with the following configuration:

::

   LLVM_BUILD_LLVM_DYLIB=On
   LLVM_LINK_LLVM_DYLIB=On

There are three shared library executable formats we're interested in: PE
Dynamic Link Library (.dll) on Windows, Mach-O Shared Object (.dylib) on Apple
systems, and ELF Shared Object (.so) on Linux, BSD and other Unix-like systems.

ELF and Mach-O Shared Object files can be built with no additional setup or
configuration. This is because all global symbols in the library are exported by
default -- the same as when building a static library. However, when building a
DLL for Windows, the situation is more complex:

- Symbols are not exported from a DLL by default. Symbols must be annotated with
  ``__declspec(dllexport)`` when building the library to be externally visible.

- Symbols imported from a Windows DLL should generally be annotated with
  ``__declspec(dllimport)`` when compiling clients.

- A single Windows DLL can export a maximum of 65,535 symbols.

Because of the requirements for Windows DLLs, additional work must be done to
ensure the proper set of public symbols is exported and visible to clients.

Annotation Macros
-----------------
The distinct DLL import and export annotations required for Windows DLLs
typically lead developers to define a preprocessor macro for annotating
exported symbols in header public files. The custom macro resolves to the
**export** annotation when building the library and the **import** annotation
when building the client.

We have defined the ``LLVM_ABI`` macro in `llvm/Support/Compiler.h
<https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/Support/Compiler.h#L152>`__
for this purpose:

.. code:: cpp

   #if defined(LLVM_EXPORTS)
   #define LLVM_ABI __declspec(dllexport)
   #else
   #define LLVM_ABI __declspec(dllimport)
   #endif

Windows DLL symbol visibility requirements are approximated on ELF and Mach-O
shared library builds by setting default symbol visibility to hidden
(``-fvisibility-default=hidden``) when building with the following
configuration:

::

   LLVM_BUILD_LLVM_DYLIB_VIS=On

For an ELF or Mach-O platform with this setting, the ``LLVM_ABI`` macro is
defined to override the default hidden symbol visibility:

.. code:: cpp

   #define LLVM_ABI __attribute__((visibility("default")))

In addition to ``LLVM_ABI``, there are a few other macros for use in less
common cases described below.

Export macros are used to annotate symbols only within their intended shared
library. This is necessary because of the way Windows handles import/export
annotations.

For example, ``LLVM_ABI`` resolves to ``__declspec(dllexport)`` only when
building source that is part of the LLVM shared library (e.g. source under
``llvm-project/llvm``). If ``LLVM_ABI`` were incorrectly used to annotate a
symbol from a different LLVM project (such as Clang) it would always resolve to
``__declspec(dllimport)`` and the symbol would not be properly exported.

How to Annotate Symbols
-----------------------
Functions
~~~~~~~~~
Exported function declarations in header files must be annotated with
``LLVM_ABI``.

.. code:: cpp

   #include "llvm/Support/Compiler.h"

   LLVM_ABI void exported_function(int a, int b);

Global Variables
~~~~~~~~~~~~~~~~
Exported global variables must be annotated with ``LLVM_ABI`` at their
``extern`` declarations.

.. code:: cpp

   #include "llvm/Support/Compiler.h"

   LLVM_ABI extern int exported_global_variable;

Classes, Structs, and Unions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Classes, structs, and unions can be annotated with ``LLVM_ABI`` at their
declaration, but this option is generally discouraged because it will
export every class member, vtable, and type information. Instead, ``LLVM_ABI``
should be applied to individual class members that require export.

In the most common case, public and protected methods without a body in the
class declaration must be annotated with ``LLVM_ABI``.

.. code:: cpp

   #include "llvm/Support/Compiler.h"

   class ExampleClass {
   public:
     // Public methods defined externally must be annotated.
     LLVM_ABI int sourceDefinedPublicMethod(int a, int b);

     // Methods defined in the class definition do not need annotation.
     int headerDefinedPublicMethod(int a, int b) {
       return a + b;
     }

     // Constructors and destructors must be annotated if defined externally.
     ExampleClass() {}
     LLVM_ABI ~ExampleClass();

     // Public static methods defined externally must be annotated.
     LLVM_ABI static int sourceDefinedPublicStaticMethod(int a, int b);
   };

Additionally, public and protected static fields that are not initialized at
declaration must be annotated with ``LLVM_ABI``.

.. code:: cpp

   #include "llvm/Support/Compiler.h"

   class ExampleClass {
   public:
     // Public static fields defined externally must be annotated.
     LLVM_ABI static int mutableStaticField;
     LLVM_ABI static const int constStaticField;

     // Static members initialized at declaration do not need to be annotated.
     static const int initializedConstStaticField = 0;
     static constexpr int initializedConstexprStaticField = 0;
   };

Private methods may also require ``LLVM_ABI`` annotation. This situation occurs
when a method defined in a header calls the private method. The private method
call may be from within the class or a friend class or method.

.. code:: cpp

   #include "llvm/Support/Compiler.h"

   class ExampleClass {
   private:
     // Private methods must be annotated if referenced by a public method defined a
     // header file.
     LLVM_ABI int privateMethod(int a, int b);

   public:
     // Inlineable method defined in the class definition calls a private method
     // defined externally. If the private method is not annotated for export, this
     // method will fail to link.
     int publicMethod(int a, int b) {
       return privateMethod(a, b);
     }
   };

There are less common cases where you may also need to annotate an inline
function even though it is fully defined in a header. Annotating an inline
function for export does not prevent it being inlined into client code. However,
it does ensure there is a single, stable address for the function exported from
the shared library.

.. code:: cpp

   #include "llvm/Support/Compiler.h"

   // Annotate the function so it is exported from the library at a fixed
   // address.
   LLVM_ABI inline int inlineFunction(int a, int b) {
     return a + b;
   }

Similarly, if a stable pointer-to-member function address is required for a
method in a C++ class, it may be annotated for export.

.. code:: cpp

   #include "llvm/Support/Compiler.h"

   class ExampleClass {
   public:
     // Annotate the method so it is exported from the library at a fixed
     // address.
     LLVM_ABI inline int inlineMethod(int a, int b) {
       return a + b;
     }
   };

.. note::

   When an inline function is annotated for export, the header containing the
   function definition **must** be included by at least one of the library's
   source files or the function will never be compiled with the export
   annotation.

Friend Functions
~~~~~~~~~~~~~~~~
Friend functions declared in a class, struct or union must be annotated with
``LLVM_ABI`` if the corresponding function declaration is annotated with
``LLVM_ABI``. This requirement applies even when the class containing the friend
declaration is annotated with ``LLVM_ABI``.

.. code:: cpp

   #include "llvm/Support/Compiler.h"

   // An exported function that has friend access to ExampleClass internals.
   LLVM_ABI int friend_function(ExampleClass &obj);

   class ExampleClass {
     // Friend declaration of a function must be annotated the same as the actual
     // function declaration.
     LLVM_ABI friend int friend_function(ExampleClass &obj);
   };

.. note::

   Annotating the friend declaration avoids an “inconsistent dll linkage”
   compiler error when building a DLL for Windows.

Virtual Table and Type Info
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Classes and structs with exported virtual methods, including child classes that
export overridden virtual methods, must also export their vtable for ELF and
Mach-O builds. This can be achieved by annotating the class rather than
individual class members.

The general rule here is to annotate at the class level if any out-of-line
method is declared ``virtual`` or ``override``.

.. code:: cpp

   #include "llvm/Support/Compiler.h"

   // Annotating the class exports vtable and type information as well as all
   // class members.
   class LLVM_ABI ParentClass {
   public:
     virtual int virtualMethod(int a, int b);
     virtual int anotherVirtualMethod(int a, int b);
     virtual ~ParentClass();
   };

   class LLVM_ABI ChildClass : public ParentClass {
   public:
     // Inline method override does not require the class be annotated.
     int virtualMethod(int a, int b) override {
       return ParentClass::virtualMethod(a, b);
     }

     // Overriding a virtual method from the parent requires the class be
     // annotated.
     int pureVirtualMethod(int a, int b) override;

     ~ChildClass();
   };

.. note::

   If a class is annotated, none of its members may be annotated. If class- and
   member-level annotations are combined on a class, it will fail compilation on
   Windows.

Compilation Errors
++++++++++++++++++
Annotating a class with ``LLVM_ABI`` causes the compiler to fully instantiate
the class at compile time. This requires exporting every method that could be
potentially used by a client even though no existing clients may actually use
them. This can cause compilation errors that were not previously present.

The most common type of error occurs when the compiler attempts to instantiate
and export a class' implicit copy constructor and copy assignment operator. If
the class contains move-only members that cannot be copied (``std::unique_ptr``
for example), the compiler will fail to instantiate these implicit
methods.

This problem is easily addressed by explicitly deleting the class' copy
constructor and copy assignment operator:

.. code:: cpp

   #include "llvm/Support/Compiler.h"

   class LLVM_ABI ExportedClass {
   public:
     ExportedClass() = default;

     // Explicitly delete the copy constructor and assignment operator.
     ExportedClass(ExportedClass const&) = delete;
     ExportedClass& operator=(ExportedClass const&) = delete;
   };

We know this modification is harmless because any clients attempting to use
these methods already would fail to compile. For a more detailed explanation,
see `this Microsoft dev blog
<https://devblogs.microsoft.com/oldnewthing/20190927-00/?p=102932>`__.

Templates
~~~~~~~~~
Most template classes are entirely header-defined and do not need to be exported
because they will be instantiated and compiled into the client as needed. Such
template classes require no export annotations. However, there are some less
common cases where annotations are required for templates.

Specialized Template Functions
++++++++++++++++++++++++++++++
As with any other exported function, an exported specialization of a template
function not defined in a header file must have its declaration annotated with
``LLVM_ABI``.

.. code:: cpp

   #include "llvm/Support/Compiler.h"

   template <typename T> T templateMethod(T a, T b) {
     return a + b;
   }

   // The explicitly specialized definition of templateMethod for int is located in
   // a source file. This declaration must be annotated with LLVM_ABI to export it.
   template <> LLVM_ABI int templateMethod(int a, int b);

Similarly, an exported specialization of a method in a template class must have
its declaration annotated with ``LLVM_ABI``.

.. code:: cpp

   #include "llvm/Support/Compiler.h"

   template <typename T> class TemplateClass {
   public:
     int method(int a, int b) {
       return a + b;
     }
   };

   // The explicitly specialized definition of method for int is defined in a
   // source file. The declaration must be annotated with LLVM_ABI to export it.
   template <> LLVM_ABI int TemplateStruct<int>::method(int a, int b);

Explicitly Instantiated Template Classes
++++++++++++++++++++++++++++++++++++++++
Explicitly instantiated template classes must be annotated with
template-specific annotations at both declaration and definition.

An extern template instantiation in a header file must be annotated with
``LLVM_TEMPLATE_ABI``. This will typically be located in a header file.

.. code:: cpp

   #include "llvm/Support/Compiler.h"

   template <typename T> class TemplateClass {
   public:
     TemplateClass(T val) : val_(val) {}

     T get() const { return val_;  }

   private:
     const T val_;
   };

   // Explicitly instantiate and export TempalateClass for int type.
   extern template class LLVM_TEMPLATE_ABI TemplateClass<int>;

The corresponding definition of the template instantiation must be annotated
with ``LLVM_EXPORT_TEMPLATE``. This will typically be located in a source file.

.. code:: cpp

   #include "TemplateClass.h"

   // Explicitly instantiate and export TempalateClass for int type.
   template class LLVM_EXPORT_TEMPLATE TemplateClass<int>;
