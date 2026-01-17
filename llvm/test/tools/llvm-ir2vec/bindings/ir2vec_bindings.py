# RUN: env PYTHONPATH=%llvm_lib_dir %python %s

import py_ir2vec

print("SUCCESS: Module imported")

# CHECK: SUCCESS: Module imported
