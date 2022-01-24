# MLIR-PyTACO: Implementing PyTACO with MLIR

TACO (http://tensor-compiler.org/) is a tensor algebra compiler. TACO defines
PyTACO, a domain specific language in Python, for writing tensor algebra
applications.

This directory contains the implementation of PyTACO using MLIR. In particular,
we implement a Python layer that accepts the PyTACO language, generates MLIR
linalg.generic OPs with sparse tensor annotation to represent the tensor
computation, and invokes the MLIR sparse tensor code generator
(https://mlir.llvm.org/docs/Dialects/SparseTensorOps/) as well as other MLIR
compilation passes to generate an executable. Then, we invoke the MLIR execution
engine to execute the program and pass the result back to the Python layer.

As can be seen from the tests in this directory, in order to port a PyTACO
program to MLIR-PyTACO, we basically only need to replace this line that imports
PyTACO:

```python
import pytaco as pt
```

with this line to import MLIR-PyTACO:

```python
from tools import mlir_pytaco_api as pt
```
