# RUN: %PYTHON %s
# Standalone sanity check of context life-cycle.
import gc
import mlir.ir

assert mlir.ir.Context._get_live_count() == 0

# Create first context.
print("CREATE C1")
c1 = mlir.ir.Context()
assert mlir.ir.Context._get_live_count() == 1
c1_repr = repr(c1)
print("C1 = ", c1_repr)

print("GETTING AGAIN...")
c2 = c1._get_context_again()
c2_repr = repr(c2)
assert mlir.ir.Context._get_live_count() == 1
assert c1_repr == c2_repr

print("C2 =", c2)

# Make sure new contexts on constructor.
print("CREATE C3")
c3 = mlir.ir.Context()
assert mlir.ir.Context._get_live_count() == 2
c3_repr = repr(c3)
print("C3 =", c3)
assert c3_repr != c1_repr
print("FREE C3")
c3 = None
gc.collect()
assert mlir.ir.Context._get_live_count() == 1

print("Free C1")
c1 = None
gc.collect()
assert mlir.ir.Context._get_live_count() == 1
print("Free C2")
c2 = None
gc.collect()
assert mlir.ir.Context._get_live_count() == 0

# Create a context, get its capsule and create from capsule.
c4 = mlir.ir.Context()
c4_capsule = c4._CAPIPtr
assert '"mlir.ir.Context._CAPIPtr"' in repr(c4_capsule)
# Because the context is already owned by Python, it cannot be created
# a second time.
try:
    c5 = mlir.ir.Context._CAPICreate(c4_capsule)
except RuntimeError:
    pass
else:
    raise AssertionError(
        "Should have gotten a RuntimeError when attempting to "
        "re-create an already owned context"
    )
c4 = None
c4_capsule = None
gc.collect()
assert mlir.ir.Context._get_live_count() == 0

# Use a private testing method to create an unowned context capsule and
# import it.
c6_capsule = mlir.ir.Context._testing_create_raw_context_capsule()
c6 = mlir.ir.Context._CAPICreate(c6_capsule)
assert mlir.ir.Context._get_live_count() == 1
c6_capsule = None
c6 = None
gc.collect()
assert mlir.ir.Context._get_live_count() == 0

# Also test operation import/export as it is tightly coupled to the context.
(
    raw_context_capsule,
    raw_operation_capsule,
) = mlir.ir.Operation._testing_create_raw_capsule("builtin.module {}")
assert '"mlir.ir.Operation._CAPIPtr"' in repr(raw_operation_capsule)
# Attempting to import an operation for an unknown context should fail.
try:
    mlir.ir.Operation._CAPICreate(raw_operation_capsule)
except RuntimeError:
    pass
else:
    raise AssertionError("Expected exception for unknown context")

# Try again having imported the context.
c7 = mlir.ir.Context._CAPICreate(raw_context_capsule)
op7 = mlir.ir.Operation._CAPICreate(raw_operation_capsule)
