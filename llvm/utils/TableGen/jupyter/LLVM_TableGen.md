# LLVM TableGen Kernel

This notebook is running `llvm-tblgen`.


```tablegen
// This is some tablegen
class Foo {}
```

    ------------- Classes -----------------
    class Foo {
    }
    ------------- Defs -----------------


Errors printed to stderr are shown.


```tablegen
This is not tablegen.
```

    <stdin>:1:1: error: Unexpected token at top level
    This is not tablegen.
    ^


Add some classes to get some output.


```tablegen
class Stuff {}
def thing : Stuff {}
```

    ------------- Classes -----------------
    class Stuff {
    }
    ------------- Defs -----------------
    def thing {	// Stuff
    }


Currently cells are not connected. Meaning that this next cell doesn't have the class from the previous one.


```tablegen
def other_thing : Stuff {}
```

    <stdin>:1:19: error: Couldn't find class 'Stuff'
    def other_thing : Stuff {}
                      ^


There is a "magic" directive `%args` that you can use to send command line arguments to `llvm-tblgen`.

For example, here we have some code that shows a warning.


```tablegen
class Thing <int A, int B> {
    int num = A;
}
```

    <stdin>:1:25: warning: unused template argument: Thing:B
    class Thing <int A, int B> {
                            ^


We can pass an argument to ignore that warning.


```tablegen
%args --no-warn-on-unused-template-args
class Thing <int A, int B> {
    int num = A;
}
```

    ------------- Classes -----------------
    class Thing<int Thing:A = ?, int Thing:B = ?> {
      int num = Thing:A;
    }
    ------------- Defs -----------------


The last `%args` in a cell will be the arguments used.


```tablegen
%args --no-warn-on-unused-template-args
%args
class Thing <int A, int B> {
    int num = A;
}
```

    <stdin>:1:25: warning: unused template argument: Thing:B
    class Thing <int A, int B> {
                            ^



```tablegen

```
