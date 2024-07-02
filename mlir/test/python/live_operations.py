# RUN: %PYTHON %s
# It is sufficient that this doesn't assert.

from mlir.ir import *


def createDetachedModule():
    module = Module.create()
    with InsertionPoint(module.body):
        # TODO: Python bindings are currently unaware that modules are also
        # operations, so having a module erased won't trigger the cascading
        # removal of live operations (#93337). Use a non-module operation
        # instead.
        nested = Operation.create("test.some_operation", regions=1)

        # When the operation is detached from parent, it is considered to be
        # owned by Python. It will therefore be erased when the Python object
        # is destroyed.
        nested.detach_from_parent()

        # However, we create and maintain references to operations within
        # `nested`. These references keep the corresponding operations in the
        # "live" list even if they have been erased in C++, making them
        # "zombie". If the C++ allocator reuses one of the address previously
        # used for a now-"zombie" operation, this used to result in an
        # assertion "cannot create detached operation that already exists" from
        # the bindings code. Erasing the detached operation should result in
        # removing all nested operations from the live list.
        #
        # Note that the assertion is not guaranteed since it depends on the
        # behavior of the allocator on the C++ side, so this test mail fail
        # intermittently.
        with InsertionPoint(nested.regions[0].blocks.append()):
            a = [Operation.create("test.some_other_operation") for i in range(100)]
    return a


def createManyDetachedModules():
    with Context() as ctx, Location.unknown():
        ctx.allow_unregistered_dialects = True
        for j in range(100):
            a = createDetachedModule()


if __name__ == "__main__":
    createManyDetachedModules()
