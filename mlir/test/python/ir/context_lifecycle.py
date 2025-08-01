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
c5 = mlir.ir.Context._CAPICreate(c4_capsule)
assert c4 is c5
c4 = None
c5 = None
gc.collect()

# Create a global threadpool and use it in two contexts
tp = mlir.ir.ThreadPool()
assert tp.get_max_concurrency() > 0
c5 = mlir.ir.Context()
c5.set_thread_pool(tp)
assert c5.get_num_threads() == tp.get_max_concurrency()
assert c5._mlir_thread_pool_ptr() == tp._mlir_thread_pool_ptr()
c6 = mlir.ir.Context()
c6.set_thread_pool(tp)
assert c6.get_num_threads() == tp.get_max_concurrency()
assert c6._mlir_thread_pool_ptr() == tp._mlir_thread_pool_ptr()
c7 = mlir.ir.Context(thread_pool=tp)
assert c7.get_num_threads() == tp.get_max_concurrency()
assert c7._mlir_thread_pool_ptr() == tp._mlir_thread_pool_ptr()
assert mlir.ir.Context._get_live_count() == 3
c5 = None
c6 = None
c7 = None
gc.collect()
