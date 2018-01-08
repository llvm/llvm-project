=====
llgoi
=====

Introduction
============

llgoi is an interactive REPL for Go. It supports expressions, statements,
most declarations and imports, including binary imports from the standard
library and source imports from ``$GOPATH``.

Example usage
=============

.. code-block:: none

  (llgo) 1+1
  #0 untyped int = 2
  (llgo) x := 1
  x untyped int = 1
  (llgo) x++
  (llgo) x
  #0 int = 2
  (llgo) import "fmt"
  (llgo) fmt.Println("hello world")
  hello world
  #0 int = 12
  #1 error (<nil>) = <nil>
  (llgo) for i := 0; i != 3; i++ {
         fmt.Println(i)
         }
  0
  1
  2
  (llgo) func foo() {
         fmt.Println("hello decl")
         }
  (llgo) foo()
  hello decl
  (llgo) import "golang.org/x/tools/go/types"
  # golang.org/x/tools/go/ast/astutil
  # golang.org/x/tools/go/exact
  # golang.org/x/tools/go/types
  (llgo) types.Eval("1+1", nil, nil)
  #0 golang.org/x/tools/go/types.TypeAndValue = {mode:4 Type:untyped int Value:2}
  #1 error (<nil>) = <nil>

Expressions
===========

Expressions can be evaluated by entering them at the llgoi prompt. The
result of evaluating the expression is displayed as if printed with the
format string ``"%+v"``. If the expression has multiple values (e.g. calls),
each value is displayed separately.

Declarations
============

Declarations introduce new entities into llgoi's scope. For example, entering
``x := 1`` introduces into the scope a variable named ``x`` with an initial
value of 1. In addition to short variable declarations (i.e. variables declared
with ``:=``), llgoi supports constant declarations, function declarations,
variable declarations and type declarations.

Imports
=======

To import a package, enter ``import`` followed by the name of a package
surrounded by quotes. This introduces the package name into llgoi's
scope. The package may be a standard library package, or a source package on
``$GOPATH``. In the latter case, llgoi will first compile the package and
its dependencies.

Statements
==========

Aside from declarations and expressions, the following kinds of statements
can be evaluated by entering them at the llgoi prompt: IncDec statements,
assignments, go statements, blocks, if statements, switch statements, select
statements and for statements.
