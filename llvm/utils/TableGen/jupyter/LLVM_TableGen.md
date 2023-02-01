# LLVM TableGen Kernel

This notebook is running `llvm-tblgen`.


```tablegen
%reset
// This is some tablegen
class Foo {}
```

    ------------- Classes -----------------
    class Foo {
    }
    ------------- Defs -----------------


Errors printed to stderr are shown.


```tablegen
%reset
This is not tablegen.
```

    <stdin>:1:1: error: Unexpected token at top level
    This is not tablegen.
    ^


Add some classes to get some output.


```tablegen
%reset
class Stuff {}
def thing : Stuff {}
```

    ------------- Classes -----------------
    class Stuff {
    }
    ------------- Defs -----------------
    def thing {	// Stuff
    }


By default cells are connected. Meaning that we cache the code and magic directives from the previously run cells.

This means that the next cell still sees the `Stuff` class.


```tablegen
def other_thing : Stuff {}
```

    ------------- Classes -----------------
    class Stuff {
    }
    ------------- Defs -----------------
    def other_thing {	// Stuff
    }
    def thing {	// Stuff
    }


You can use the magic `%reset` to clear this cache and start fresh.


```tablegen
%reset
def other_thing : Stuff {}
```

    <stdin>:1:19: error: Couldn't find class 'Stuff'
    def other_thing : Stuff {}
                      ^


There is a "magic" directive `%args` that you can use to send command line arguments to `llvm-tblgen`.

For example, here we have some code that shows a warning.


```tablegen
%reset
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
```

    ------------- Classes -----------------
    class Thing<int Thing:A = ?, int Thing:B = ?> {
      int num = Thing:A;
    }
    ------------- Defs -----------------


If you have a run of cells without a `%reset`, the most recent `%args` is used.


```tablegen
// This passes --no-warn-on-unused-template-args
```

    ------------- Classes -----------------
    class Thing<int Thing:A = ?, int Thing:B = ?> {
      int num = Thing:A;
    }
    ------------- Defs -----------------



```tablegen
%args
// Now we're not passing the argument so the warning comes back.
```

    <stdin>:1:25: warning: unused template argument: Thing:B
    class Thing <int A, int B> {
                            ^


If there are many `%args` in a cell, the last one is used.


```tablegen
%reset
%args --no-warn-on-unused-template-args
%args
class Thing <int A, int B> {}
```

    <stdin>:1:18: warning: unused template argument: Thing:A
    class Thing <int A, int B> {}
                     ^
    <stdin>:1:25: warning: unused template argument: Thing:B
    class Thing <int A, int B> {}
                            ^

