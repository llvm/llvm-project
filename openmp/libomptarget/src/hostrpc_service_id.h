/*
 *    hostrpc_service_id.h:  This header contains the enum for all the
 *                           implemented services in hostcall.  This header is
 *                           included by both device stubs and host routines.
 *                           It also includes the version, release, and patch
 *                           identification for hostcall.

MIT License

Copyright © 2019 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished
to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/

#ifndef __HOSTCALL_SERVICE_ID_H__
#define __HOSTCALL_SERVICE_ID_H__

// Please update at least the patch level when adding a new service.
// This will ensure that applications that use a new device stub do not
// try to use backlevel hostcall host runtimes that do not have the
// implmented host version of the service.
//
#define HOSTCALL_VERSION 0
#define HOSTCALL_RELEASE 6
#define HOSTCALL_PATCH 3
// HOSTCALL_VRM fits in two bytes allowing for 64 patches, 64 releases, and 15
// versions
#define HOSTCALL_VRM                                                           \
  ((HOSTCALL_VERSION * 4096) + (HOSTCALL_RELEASE * 64) + HOSTCALL_PATCH)
#define HOSTCALL_VERSION_RELEASE ((HOSTCALL_VERSION * 64) + HOSTCALL_RELEASE)
typedef short hostcall_version_t;

#define PACK_VERS(x) ((uint32_t)HOSTCALL_VRM << 16) | ((uint32_t)x)

enum hostcall_service_id {
  HOSTCALL_SERVICE_UNUSED,
  HOSTCALL_SERVICE_TERMINATE,
  HOSTCALL_SERVICE_PRINTF,
  HOSTCALL_SERVICE_MALLOC,
  HOSTCALL_SERVICE_MALLOC_PRINTF,
  HOSTCALL_SERVICE_FREE,
  HOSTCALL_SERVICE_DEMO,
  HOSTCALL_SERVICE_FUNCTIONCALL,
  HOSTCALL_SERVICE_VARFNUINT,
  HOSTCALL_SERVICE_VARFNUINT64,
  HOSTCALL_SERVICE_VARFNDOUBLE,
};
typedef enum hostcall_service_id hostcall_service_id_t;

// Services execute on the host as either SERIAL or PARALLEL.
// There are typically multiple host consumer threads running in parallel.
// If a service is SERIAL, the consumer threads will queue each
// service request for a single (serial) pthread to execute. If a service
// request is parallel, the host consumer threads execute the
// service without waiting for the serial pthread.
//
enum hostcall_type {
  HOSTCALL_SERIAL,
  HOSTCALL_PARALLEL,
};
typedef enum hostcall_type hostcall_type_t;

#endif // __HOSTCALL_SERVICE_ID_H__
