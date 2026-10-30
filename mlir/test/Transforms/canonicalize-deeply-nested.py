# RUN: %PYTHON %s | mlir-opt -canonicalize | FileCheck %s --implicit-check-not=arith.addi

# Identity folds change values used in nested regions. The store keeps
# all loops live, so repeated deadness queries must account for nested effects.
# Generate the input to keep the test small. Do not check elapsed time.

# CHECK-LABEL: func.func @deeply_nested(
# CHECK-SAME: %[[VALUE:[^ ,]+]]: i32, %[[OUT:[^ ,]+]]: memref<i32>
# CHECK-COUNT-1000: scf.for
# CHECK-NEXT: memref.store %[[VALUE]], %[[OUT]][] : memref<i32>
# CHECK-NOT: scf.for

depth = 1000
print("func.func @deeply_nested(%value: i32, %out: memref<i32>) {")
print("%c0 = arith.constant 0 : index")
print("%c1 = arith.constant 1 : index")
print("%c4 = arith.constant 4 : index")
print("%zero = arith.constant 0 : i32")
value = "%value"
for i in range(depth):
    print(f"scf.for %i{i} = %c0 to %c4 step %c1 {{")
    print(f"%v{i} = arith.addi {value}, %zero : i32")
    value = f"%v{i}"
print(f"memref.store {value}, %out[] : memref<i32>")
for _ in range(depth):
    print("}")
print("return")
print("}")
