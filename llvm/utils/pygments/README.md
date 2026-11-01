# LLVM IR Pygments lexer

`llvm_lexer.py` extends the LLVM IR lexer that ships with Pygments. It adds the
keywords accepted by `llvm/lib/AsmParser/LLLexer.cpp` and the attributes listed
in `llvm/include/llvm/IR/Attributes.td`. Both files are read when the lexer is
loaded, so it stays in sync with the IR.

The Sphinx docs register it through
`utils/docs/llvm_sphinx/ext/llvm_pygments.py`.

To use it from the command line, run this from the llvm-project root:

```
pygmentize -x -l llvm/utils/pygments/llvm_lexer.py:LlvmLexer foo.ll
```
