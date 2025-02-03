def : Function {
    let name = "olCreateKernel";
    let desc = "";
    let details = [];
    let params = [
        Param<"ol_program_handle_t", "Program", "handle of the program", PARAM_IN>,
        Param<"const char*", "KernelName", "name of the kernel entry point in the program", PARAM_IN>,
        Param<"ol_kernel_handle_t*", "Kernel", "output pointer for the created kernel", PARAM_OUT>
    ];
    let returns = [];
}

def : Function {
    let name = "olRetainKernel";
    let desc = "Increment the reference count of the given kernel";
    let details = [];
    let params = [
        Param<"ol_kernel_handle_t", "Kernel", "handle of the kernel", PARAM_IN>
    ];
    let returns = [];
}

def : Function {
    let name = "olReleaseKernel";
    let desc = "Decrement the reference count of the given kernel";
    let details = [];
    let params = [
        Param<"ol_kernel_handle_t", "Kernel", "handle of the kernel", PARAM_IN>
    ];
    let returns = [];
}

def : Function {
    let name = "olSetKernelArgValue";
    let desc = "Set the value of a single kernel argument at the given index";
    let details = [
        "The implementation will construct and lay out the backing storage for the kernel arguments."
        "The effects of calls to this function on a kernel are lost if olSetKernelArgsData is called."
    ];
    let params = [
        Param<"ol_kernel_handle_t", "Kernel", "handle of the kernel", PARAM_IN>,
        Param<"uint32_t", "Index", "index of the argument", PARAM_IN>,
        Param<"size_t", "Size", "size of the argument data", PARAM_IN>,
        Param<"void*", "ArgData", "pointer to the argument data", PARAM_IN>
    ];
    let returns = [];
}

def : Function {
    let name = "olSetKernelArgsData";
    let desc = "Set the entire argument data for a kernel";
    let details = [
        "Previous calls to olSetKernelArgValue on the same kernel are invalidated by this function"
        "The data pointed to by ArgsData is assumed to be laid out correctly according to the requirements of the backend API"
    ];
    let params = [
        Param<"ol_kernel_handle_t", "Kernel", "handle of the kernel", PARAM_IN>,
        Param<"void*", "ArgsData", "pointer to the argument data", PARAM_IN>,
        Param<"size_t", "ArgsDataSize", "size of the argument data", PARAM_IN>
    ];
    let returns = [];
}
