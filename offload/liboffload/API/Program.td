//===-- Program.td - Program definitions for Offload -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains Offload API definitions related to the program handle
//
//===----------------------------------------------------------------------===//

def olCreateProgram : Function {
    let desc = "Create a program for the device from the binary image pointed to by `ProgData` within the given context.";
    let details = [
        "The provided `ProgData` will be copied and need not outlive the returned handle",
        "The program is scoped to `Context` and `Device` must belong to it.",
        "The program can only be used with queues that were created in the same context."
    ];
    let params = [
        Param<"ol_context_handle_t", "Context", "handle of the context", PARAM_IN>,
        Param<"ol_device_handle_t", "Device", "handle of the device", PARAM_IN>,
        Param<"const void*", "ProgData", "pointer to the program binary data", PARAM_IN>,
        Param<"size_t", "ProgDataSize", "size of the program binary in bytes", PARAM_IN>,
        Param<"ol_program_handle_t*", "Program", "output pointer for the created program", PARAM_OUT>
    ];
    let returns = [
        Return<"OL_ERRC_INVALID_BINARY", [
            "If the buffer described by `ProgData` and `ProgDataSize` is not a valid binary image for the platform."
        ]>,
        Return<"OL_ERRC_INVALID_DEVICE", [
            "Device does not belong to `Context`"
        ]>,
    ];
}

def olIsValidBinary : Function {
    let desc = "Validate if the binary image pointed to by `ProgData` is compatible with the device.";
    let details = ["The provided `ProgData` will not be loaded onto the device"];
    let params = [
        Param<"ol_device_handle_t", "Device", "handle of the device", PARAM_IN>,
        Param<"const void*", "ProgData", "pointer to the program binary data", PARAM_IN>,
        Param<"size_t", "ProgDataSize", "size of the program binary in bytes", PARAM_IN>,
        Param<"bool*", "Valid", "output is true if the image is compatible", PARAM_OUT>
    ];
    let returns = [];
}

def olIterateCompatibleDevices : Function {
    let desc = "Iterates over all available devices that are compatible with the binary image pointed to by `ProgData`, calling the callback for each device.";
    let details = [
        "The provided `ProgData` will not be loaded onto any device",
        "If the user-provided callback returns `false`, the iteration is stopped.",
        "Only platforms that are compatible with the binary image will be initialized"
    ];
    let params = [
        Param<"const void*", "ProgData", "pointer to the program binary data", PARAM_IN>,
        Param<"size_t", "ProgDataSize", "size of the program binary in bytes", PARAM_IN>,
        Param<"ol_device_iterate_cb_t", "Callback", "user-provided function called for each compatible device", PARAM_IN>,
        Param<"void*", "UserData", "optional user data to pass to the callback", PARAM_IN_OPTIONAL>
    ];
    let returns = [];
}

def olDestroyProgram : Function {
    let desc = "Destroy the program and free all underlying resources.";
    let details = [];
    let params = [
        Param<"ol_program_handle_t", "Program", "handle of the program", PARAM_IN>
    ];
    let returns = [];
}

def ol_program_info_t : Enum {
  let desc = "Supported program info.";
  let is_typed = 1;
  let etors = [
    TaggedEtor<"IMAGE_BASE", "void *", "Start of the host-side image loaded for this program. For IR inputs that were compiled by the platform this is the compiled image.">,
    TaggedEtor<"IMAGE_SIZE", "size_t", "Size in bytes of the image described by `OL_PROGRAM_INFO_IMAGE_BASE`.">,
  ];
}

def olGetProgramInfo : Function {
  let desc = "Queries the given property of the program.";
  let details = [
    "`olGetProgramInfoSize` can be used to query the storage size "
    "required for the given query.",
    "Returned image pointers are owned by the program and remain valid until it is destroyed."
  ];
  let params = [
    Param<"ol_program_handle_t", "Program", "handle of the program", PARAM_IN>,
    Param<"ol_program_info_t", "PropName", "type of the info to retrieve", PARAM_IN>,
    Param<"size_t", "PropSize", "the number of bytes pointed to by PropValue.", PARAM_IN>,
    TypeTaggedParam<"void*", "PropValue", "array of bytes holding the info. "
      "If PropSize is not equal to or greater to the real number of bytes needed to return the info "
      "then the OL_ERRC_INVALID_SIZE error is returned and PropValue is not used.", PARAM_OUT,
      TypeInfo<"PropName" , "PropSize">>
  ];
  let returns = [
    Return<"OL_ERRC_INVALID_SIZE", [
      "`PropSize == 0`",
      "If `PropSize` is less than the real number of bytes needed to return the info."
    ]>,
    Return<"OL_ERRC_INVALID_NULL_HANDLE">
  ];
}

def olGetProgramInfoSize : Function {
  let desc = "Returns the storage size of the given program query.";
  let details = [];
  let params = [
    Param<"ol_program_handle_t", "Program", "handle of the program", PARAM_IN>,
    Param<"ol_program_info_t", "PropName", "type of the info to query", PARAM_IN>,
    Param<"size_t*", "PropSizeRet", "pointer to the number of bytes required to store the query", PARAM_OUT>
  ];
  let returns = [
    Return<"OL_ERRC_INVALID_NULL_HANDLE">
  ];
}
