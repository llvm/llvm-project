# Chapter 1: Toy Language and AST

[TOC]

## The Language

This tutorial will be illustrated with a toy language that we’ll call “Toy”
(naming is hard...). Toy is a tensor-based language that allows you to define
functions, perform some math computation, and print results.

Given that we want to keep things simple, the codegen will be limited to tensors
of rank <= 2, and the only datatype in Toy is a 64-bit floating point type (aka
‘double’ in C parlance). As such, all values are implicitly double precision,
`Values` are immutable (i.e. every operation returns a newly allocated value),
and deallocation is automatically managed. But enough with the long description;
nothing is better than walking through an example to get a better understanding:

```toy
def main() {
  # Define a variable `a` with shape <2, 3>, initialized with the literal value.
  # The shape is inferred from the supplied literal.
  var a = [[1, 2, 3], [4, 5, 6]];

  # b is identical to a, the literal tensor is implicitly reshaped: defining new
  # variables is the way to reshape tensors (element count must match).
  var b<2, 3> = [1, 2, 3, 4, 5, 6];

  # transpose() and print() are the only builtin, the following will transpose
  # a and b and perform an element-wise multiplication before printing the result.
  print(transpose(a) * transpose(b));
}
```

Type checking is statically performed through type inference; the language only
requires type declarations to specify tensor shapes when needed. Functions are
generic: their parameters are unranked (in other words, we know these are
tensors, but we don't know their dimensions). They are specialized for every
newly discovered signature at call sites. Let's revisit the previous example by
adding a user-defined function:

```toy
# User defined generic function that operates on unknown shaped arguments.
def multiply_transpose(a, b) {
  return transpose(a) * transpose(b);
}

def main() {
  # Define a variable `a` with shape <2, 3>, initialized with the literal value.
  var a = [[1, 2, 3], [4, 5, 6]];
  var b<2, 3> = [1, 2, 3, 4, 5, 6];

  # This call will specialize `multiply_transpose` with <2, 3> for both
  # arguments and deduce a return type of <3, 2> in initialization of `c`.
  var c = multiply_transpose(a, b);

  # A second call to `multiply_transpose` with <2, 3> for both arguments will
  # reuse the previously specialized and inferred version and return <3, 2>.
  var d = multiply_transpose(b, a);

  # A new call with <3, 2> (instead of <2, 3>) for both dimensions will
  # trigger another specialization of `multiply_transpose`.
  var e = multiply_transpose(c, d);

  # Finally, calling into `multiply_transpose` with incompatible shape will
  # trigger a shape inference error.
  var f = multiply_transpose(transpose(a), c);
}
```

## The AST

The AST from the above code is fairly straightforward; here is a dump of it:

```
Module:
  Function 
    Proto 'multiply_transpose' @test/Examples/Toy/Ch1/ast.toy:4:1'
    Params: [a, b]
    Block {
      Return
        BinOp: * @test/Examples/Toy/Ch1/ast.toy:5:25
          Call 'transpose' [ @test/Examples/Toy/Ch1/ast.toy:5:10
            var: a @test/Examples/Toy/Ch1/ast.toy:5:20
          ]
          Call 'transpose' [ @test/Examples/Toy/Ch1/ast.toy:5:25
            var: b @test/Examples/Toy/Ch1/ast.toy:5:35
          ]
    } // Block
  Function 
    Proto 'main' @test/Examples/Toy/Ch1/ast.toy:8:1'
    Params: []
    Block {
      VarDecl a<> @test/Examples/Toy/Ch1/ast.toy:11:3
        Literal: <2, 3>[ <3>[ 1.000000e+00, 2.000000e+00, 3.000000e+00], <3>[ 4.000000e+00, 5.000000e+00, 6.000000e+00]] @test/Examples/Toy/Ch1/ast.toy:11:11
      VarDecl b<2, 3> @test/Examples/Toy/Ch1/ast.toy:15:3
        Literal: <6>[ 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00] @test/Examples/Toy/Ch1/ast.toy:15:17
      VarDecl c<> @test/Examples/Toy/Ch1/ast.toy:19:3
        Call 'multiply_transpose' [ @test/Examples/Toy/Ch1/ast.toy:19:11
          var: a @test/Examples/Toy/Ch1/ast.toy:19:30
          var: b @test/Examples/Toy/Ch1/ast.toy:19:33
        ]
      VarDecl d<> @test/Examples/Toy/Ch1/ast.toy:22:3
        Call 'multiply_transpose' [ @test/Examples/Toy/Ch1/ast.toy:22:11
          var: b @test/Examples/Toy/Ch1/ast.toy:22:30
          var: a @test/Examples/Toy/Ch1/ast.toy:22:33
        ]
      VarDecl e<> @test/Examples/Toy/Ch1/ast.toy:25:3
        Call 'multiply_transpose' [ @test/Examples/Toy/Ch1/ast.toy:25:11
          var: b @test/Examples/Toy/Ch1/ast.toy:25:30
          var: c @test/Examples/Toy/Ch1/ast.toy:25:33
        ]
      VarDecl f<> @test/Examples/Toy/Ch1/ast.toy:28:3
        Call 'multiply_transpose' [ @test/Examples/Toy/Ch1/ast.toy:28:11
          Call 'transpose' [ @test/Examples/Toy/Ch1/ast.toy:28:30
            var: a @test/Examples/Toy/Ch1/ast.toy:28:40
          ]
          var: c @test/Examples/Toy/Ch1/ast.toy:28:44
        ]
    } // Block
```

You can reproduce this result and play with the example in the
`examples/toy/Ch1/` directory; try running `path/to/BUILD/bin/toyc-ch1
test/Examples/Toy/Ch1/ast.toy -emit=ast`.

The code for the lexer is fairly straightforward; it is all in a single header:
`examples/toy/Ch1/include/toy/Lexer.h`. The parser can be found in
`examples/toy/Ch1/include/toy/Parser.h`; it is a recursive descent parser. If
you are not familiar with such a Lexer/Parser, these are very similar to the
LLVM Kaleidoscope equivalent that are detailed in the first two chapters of the
[Kaleidoscope Tutorial](https://llvm.org/docs/tutorial/MyFirstLanguageFrontend/LangImpl02.html).

The [next chapter](Ch-2.md) will demonstrate how to convert this AST into MLIR.
