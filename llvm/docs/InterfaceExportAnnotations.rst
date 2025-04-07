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

For ELF and Mach-O builds, this configuration works as-is because all symbols
are exported by default. To build a Windows DLL, however, the situation is more
complex:

- Symbols are not exported from a DLL by default. Symbols must be annotated with
  ``__declspec(dllexport)`` when building the library to be externally visible.

- Symbols imported from a Windows DLL should generally be annotated with
  ``__declspec(dllimport)`` when compiling clients.

- A single Windows DLL can export a maximum of 65,535 symbols.

Annotation Macros
-----------------
The distinct DLL import and export annotations required for Windows DLLs
typically lead developers to define a preprocessor macro for annotating exported
symbols in header public files. The custom macro resolves to the _export_
annotation when building the library and the _import_ annotation when building
the client.

For this purpose, we have defined the `LLVM_ABI` macro in
`llvm/Support/Compiler.h
<https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/Support/Compiler.h#L152>`__.

.. code:: cpp

   #if defined(LLVM_EXPORTS)
   #define LLVM_ABI __declspec(dllexport)
   #else
   #define LLVM_ABI __declspec(dllimport)
   #endif

Because building LLVM for Windows is less common than ELF and Mach-O platforms,
Windows DLL symbol visibility requirements are approximated on these platforms
by setting default symbol visibility to hidden
(``-fvisibility-default=hidden``) when building with the following
configuration:

::

   LLVM_BUILD_LLVM_DYLIB_VIS=On

For an ELF or Mach-O platform with this setting, the ``LLVM_ABI`` macro is
defined to override the default hidden symbol visibility:

.. code:: cpp

   #define LLVM_ABI __attribute__((visibility("default")))

In addition to ``LLVM_ABI``, there are a few other macros for use in less
common cases described below. All ``LLVM_`` export macros are only for use with
symbols defined in the LLVM library. They must not be used to annotate symbols
defined in other LLVM projects such as lld, lldb, and clang. Their use should
generally restricted to source code under ``llvm-project/llvm``.

Annotating Symbols
------------------

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
definition, but this option is generally discouraged. Annotating the entire
definition exports unnecessary symbols, such as private functions, vtables, and
type information. Instead, ``LLVM_ABI`` should be applied only to public and
protected method declarations without a body in the header, including
constructors and destructors.

.. code:: cpp

   #include "llvm/Support/Compiler.h"

   class ExampleClass {
   public:
     // Public methods defined externally must be annotatated.
     LLVM_ABI int sourceDefinedPublicMethod(int a, int b);

     // Methods defined in the class definition do not need annotation.
     int headerDefinedPublicMethod(int a, int b) {
       return a + b;
     }

     // Constructors and destructors must be annotated if defined externally.
     ExampleClass() {}
     LLVM_ABI ~ExampleClass();

     // Public static methods defined externally must be annotatated.
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

Private methods may also require ``LLVM_ABI`` annotation in certain cases. This
situation occurs when a method defined in a header calls the private method. The
private method call may be from within the class, a parent class, or a friend
class.

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

Friend Functions
~~~~~~~~~~~~~~~~

Friend functions declared in a class, struct or union must be annotated with
``LLVM_ABI`` if the corresponding function declaration is also annotated. This
requirement applies even when the class itself is annotated with ``LLVM_ABI``.

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
   compiler error when building for Windows. This annotation is harmless but not
   required when building ELF or Mach-O shared libraries.

VTable and Type Info
~~~~~~~~~~~~~~~~~~~~

Classes and structs with exported virtual methods, or child classes that export
overridden virtual methods, must also export their vtable for ELF and Mach-O
builds. This can be achieved by annotating the class rather than individual
class members.

.. code:: cpp

   #include "llvm/Support/Compiler.h"

   class ParentClass {
   public:
     virtual int virtualMethod(int a, int b);
     virtual int anotherVirtualMethod(int a, int b);
     virtual ~ParentClass();
   };

   // Annotating the class exports vtable and type information as well as all
   // class members.
   class LLVM_ABI ChildClass : public ParentClass {
   public:
     // Inline method override does not require the class be annotated.
     int virtualMethod(int a, int b) override {
       return ParentClass::virtualMethod(a, b);
     }

     // Overriding a virtual method from the parent requires the class be
     // annotated. The parent class may require annotation as well.
     int pureVirtualMethod(int a, int b) override;
     ~ChildClass();
   };

If annotating a type with ``LLVM_ABI`` causes compilation issues such as those
described
`here <https://devblogs.microsoft.com/oldnewthing/20190927-00/?p=102932>`__,
the class may require modification. Often, explicitly deleting the copy
constructor and copy assignment operator will resolve the issue.

.. code:: cpp

   #include "llvm/Support/Compiler.h"

   #include <vector>

   class LLVM_ABI ExportedClass {
   public:
     // Explicitly delete the copy constructor and assignment operator.
     ExportedClass(ExportedClass const&) = delete;
     ExportedClass& operator=(ExportedClass const&) = delete;
   };

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
