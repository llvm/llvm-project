//===-- AdbClientUtils.h --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLDB_SOURCE_PLUGINS_PLATFORM_ANDROID_ADBCLIENTUTILS_H
#define LLDB_SOURCE_PLUGINS_PLATFORM_ANDROID_ADBCLIENTUTILS_H

#include "lldb/Utility/Status.h"
#include <string>

namespace lldb_private {
class Connection;

namespace platform_android {

const std::chrono::seconds kReadTimeout(20);
const static char *kOKAY = "OKAY";
const static char *kFAIL = "FAIL";
const static char *kDATA = "DATA";
const static char *kDONE = "DONE";
const static char *kSEND = "SEND";
const static char *kRECV = "RECV";
const static char *kSTAT = "STAT";
const static size_t kSyncPacketLen = 8;
const static size_t kMaxPushData = 2 * 1024;
const static uint32_t kDefaultMode = 0100770;

namespace adb_client_utils {

Status ReadAllBytes(Connection &conn, void *buffer, size_t size);
Status SendAdbMessage(Connection &conn, const std::string &packet);
Status GetResponseError(Connection &conn, const char *response_id);
Status ConnectToAdb(Connection &conn);
Status EnterSyncMode(Connection &conn);
Status SelectTargetDevice(Connection &conn, const std::string &device_id);
Status ReadAdbMessage(Connection &conn, std::vector<char> &message);
Status ReadResponseStatus(Connection &conn);

} // namespace adb_client_utils
} // namespace platform_android
} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_PLATFORM_ANDROID_ADBCLIENTUTILS_H
