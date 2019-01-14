//==-- get_device_count_by_type.cpp - Get device count by type -------------==//
//
// The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include <vector>
#include <string>
#include <CL/cl.h>
#include <CL/cl_ext.h>

static const std::string help =
"   Help\n"
"   Example: ./get_device_count_by_type cpu\n"
"   Support types: cpu/gpu/accelerator/default/all\n"
"   Output format: <number_of_devices>:<additional_Information>";

int main(int argc, char* argv[]) {
    if (argc <= 1) {
        std::cout << "0:Please set a device type for find" << std::endl
            << help << std::endl;
        return 0;
    }

    std::string type = argv[1];
    cl_device_type device_type;
    if (type == "cpu") {
        device_type = CL_DEVICE_TYPE_CPU;
    } else if (type == "gpu") {
        device_type = CL_DEVICE_TYPE_GPU;
    } else if (type == "accelerator") {
        device_type = CL_DEVICE_TYPE_ACCELERATOR;
    } else if (type == "default") {
        device_type = CL_DEVICE_TYPE_DEFAULT;
    } else if (type == "all") {
        device_type = CL_DEVICE_TYPE_ALL;
    } else  {
        std::cout << "0:Incorrect device type." << std::endl
            << help << std::endl;
        return 0;
    }

    cl_int iRet = CL_SUCCESS;
    cl_uint platformCount = 0;

    iRet = clGetPlatformIDs(0, nullptr, &platformCount);
    if (iRet != CL_SUCCESS) {
        if (iRet == CL_PLATFORM_NOT_FOUND_KHR) {
            std::cout << "0:OpenCL runtime not found " << std::endl;
        } else {
            std::cout << "0:A problem at calling function clGetPlatformIDs count "
                << iRet << std::endl;
        }
        return 0;
    }

    std::vector<cl_platform_id> platforms(platformCount);

    iRet = clGetPlatformIDs(platformCount, &platforms[0], nullptr);
    if (iRet != CL_SUCCESS) {
        std::cout << "0:A problem at when calling function clGetPlatformIDs ids " << iRet << std::endl;
        return 0;
    }

    cl_uint deviceCount = 0;
    for (cl_uint i = 0; i < platformCount; i++) {
        cl_uint deviceCountPart = 0;
        iRet = clGetDeviceIDs(platforms[i], device_type, 0, nullptr, &deviceCountPart);
        if (iRet == CL_SUCCESS) {
            deviceCount += deviceCountPart;
        }
    }

    std::cout << deviceCount << ":" << std::endl;
    return 0;
}
