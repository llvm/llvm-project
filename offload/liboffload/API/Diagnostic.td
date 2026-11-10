//===-- Diagnostic.td - Diagnostic definitions for Offload -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Diagnostics reporting is split into two halves:
//
//  - Detection lives in the plugins. It notices the problem, triages it, and
//    gathers everything worth reporting into a self-contained diagnostic. It
//    also judges whether the problem is fatal, which the handler can override.
//
//  - Messaging turns a diagnostic into user-visible text. It is configurable by
//    the users of the library, which set a handler on each device.
//
//===----------------------------------------------------------------------===//

def ol_diagnostic_kind_t : Enum {
  let desc = "The kind of problem that was detected";
  let etors = [
    Etor<"DEALLOCATION_OF_NON_ALLOCATED_PTR", "Deallocation of a pointer that was never allocated">,
    Etor<"DOUBLE_FREE", "Deallocation of a pointer that was already deallocated">,
    Etor<"DEALLOCATION_OF_WRONG_PTR_KIND", "Deallocation with a different allocation kind than the allocation used">,
    Etor<"MEMORY_ACCESS_FAULT", "The device faulted accessing memory">,
    Etor<"TRAP_IN_KERNEL", "A kernel executed a trap instruction">,
  ];
}

def ol_pointer_relation_t : Enum {
  let desc = "Where a device pointer sits with respect to the known allocations";
  let etors = [
    Etor<"NO_ALLOCATIONS", "No host-issued allocation is known at all">,
    Etor<"INSIDE", "The pointer points into a live allocation">,
    Etor<"INSIDE_PRIOR", "The pointer points into an allocation that was freed">,
    Etor<"NEAR", "The pointer is outside of, but close to, a known allocation">,
    Etor<"FAR", "The pointer is far away from any known allocation">,
  ];
}

def ol_allocation_info_t : Struct {
  let desc = "Information about a tracked allocation, and about the allocation history at the same base pointer";
  let members = [
    StructMember<"const char*", "AllocationTrace", "The stack trace of the allocation itself. Never null, possibly empty">,
    StructMember<"const char*", "DeallocationTrace", "The stack trace of the deallocation. Never null, empty if still live">,
    StructMember<"void*", "DevicePtr", "The allocated device pointer">,
    StructMember<"void*", "HostPtr", "The corresponding host pointer, or null">,
    StructMember<"uint64_t", "Size", "The size of the allocation">,
    StructMember<"const struct ol_allocation_info_t*", "Prior", "Prior allocations at the same base pointer, most recent first. Null when NumPrior is zero">,
    StructMember<"size_t", "NumPrior", "The number of entries in Prior">,
  ];
}

def ol_kernel_launch_info_t : Struct {
  let desc = "Information about a tracked kernel launch";
  let members = [
    StructMember<"const char*", "Name", "The symbol name of the kernel, not prettified. Never null">,
    StructMember<"const char*", "LaunchTrace", "The stack trace of the launch. Never null, empty if it was not tracked">,
    StructMember<"int", "IsCulprit", "Non-zero if this launch was identified as the one that caused the problem">,
  ];
}

def ol_pointer_info_t : Struct {
  let desc = "Information about the device pointer a diagnostic is about";
  let members = [
    StructMember<"void*", "DevicePtr", "The device pointer in question">,
    StructMember<"ol_pointer_relation_t", "Relation", "How DevicePtr relates to the known allocations">,
    StructMember<"uintptr_t", "Distance", "The distance to Allocation in bytes, meaningful for OL_POINTER_RELATION_NEAR">,
    StructMember<"const struct ol_allocation_info_t*", "Allocation", "The allocation Relation refers to, or null if there is none">,
  ];
}

def ol_diagnostic_hints_t : Struct {
  let desc = "What is being tracked, so that a handler can suggest how to get more information without knowing about environment variables";
  let members = [
    StructMember<"int", "AllocationTracesEnabled", "Non-zero if allocation stack traces are being tracked">,
    StructMember<"const char*", "AllocationTracesEnvar", "The name of the environment variable controlling the above">,
    StructMember<"const char*", "KernelLaunchesEnvar", "The name of the environment variable controlling how many kernel launch traces are tracked">,
    StructMember<"int", "KernelLaunchesKnown", "Non-zero if information about kernel launches was available at all. If zero, no kernel launches should be reported, not even their absence">,
    StructMember<"uint32_t", "KernelLaunchesRequested", "The number of kernel launch traces that were requested">,
    StructMember<"uint32_t", "KernelLaunchesAvailable", "The number of kernel launches known">,
    StructMember<"uint64_t", "KernelLaunchesCapacity", "The maximal number of kernel launches that can be known">,
  ];
}

def ol_diagnostic_t : Struct {
  let desc = "A self-contained description of a detected problem";
  let members = [
    StructMember<"ol_diagnostic_kind_t", "Kind", "The kind of problem">,
    StructMember<"int", "HasDeviceId", "Non-zero if DeviceId is known">,
    StructMember<"int32_t", "DeviceId", "The device the problem was detected on, if HasDeviceId is non-zero">,
    StructMember<"const char*", "Summary", "A one-line description of the problem. Never null">,
    StructMember<"const char*", "OffendingStackTrace", "The host stack trace at the point of detection. Never null, possibly empty">,
    StructMember<"const struct ol_pointer_info_t*", "Pointer", "The device pointer involved, or null if none is">,
    StructMember<"const struct ol_kernel_launch_info_t*", "KernelTraces", "The kernel launches worth reporting, most recent first. Null when NumKernelTraces is zero">,
    StructMember<"size_t", "NumKernelTraces", "The number of entries in KernelTraces">,
    StructMember<"ol_diagnostic_hints_t", "Tracking", "What is being tracked, for \"use X to get more information\" hints">,
  ];
}

def ol_diagnostic_response_t : Struct {
  let desc = "What a handler did with a diagnostic. The detection side pre-fills it with the defaults, the handler only has to touch what it wants to change";
  let members = [
    StructMember<"int", "Reported", "Non-zero if the handler reported the diagnostic. Defaults to zero, in which case the minimal built-in reporting is printed. The execution ends after the handler returns either way">,
  ];
}

def ol_diagnostic_cb_t : FptrTypedef {
  let desc = "Turns a diagnostic into a user-visible message and says in `Response` whether it did so. The execution ends once the callback returns: the reported problems are not recoverable. Must be thread safe, it can be called from plugin-owned threads.";
  let params = [
    Param<"const struct ol_diagnostic_t*", "Diag", "the detected problem", PARAM_IN>,
    Param<"void*", "UserData", "the user data of the handler", PARAM_IN_OPTIONAL>,
    Param<"struct ol_diagnostic_response_t*", "Response", "what the handler did, pre-filled with the defaults", !or(PARAM_IN, PARAM_OUT)>
  ];
  let return = "void";
}

def ol_diagnostic_handler_t : Struct {
  let desc = "A diagnostic handler: what to call, and what to pass it";
  let members = [
    StructMember<"ol_diagnostic_cb_t", "Callback", "The function called for each diagnostic">,
    StructMember<"void*", "UserData", "Passed to Callback">,
  ];
}
