//===-- ActivityStreamSPI.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_DEBUGSERVER_SOURCE_MACOSX_DARWINLOG_ACTIVITYSTREAMSPI_H
#define LLDB_TOOLS_DEBUGSERVER_SOURCE_MACOSX_DARWINLOG_ACTIVITYSTREAMSPI_H

#include <sys/time.h>
#include <xpc/xpc.h>

#define OS_ACTIVITY_MAX_CALLSTACK 32

// Enums

enum {
  OS_ACTIVITY_STREAM_PROCESS_ONLY = 0x00000001,
  OS_ACTIVITY_STREAM_SKIP_DECODE = 0x00000002,
  OS_ACTIVITY_STREAM_PAYLOAD = 0x00000004,
  OS_ACTIVITY_STREAM_HISTORICAL = 0x00000008,
  OS_ACTIVITY_STREAM_CALLSTACK = 0x00000010,
  OS_ACTIVITY_STREAM_DEBUG = 0x00000020,
  OS_ACTIVITY_STREAM_BUFFERED = 0x00000040,
  OS_ACTIVITY_STREAM_NO_SENSITIVE = 0x00000080,
  OS_ACTIVITY_STREAM_INFO = 0x00000100,
  OS_ACTIVITY_STREAM_PROMISCUOUS = 0x00000200,
  OS_ACTIVITY_STREAM_PRECISE_TIMESTAMPS = 0x00000200
};
typedef uint32_t os_activity_stream_flag_t;

enum {
  OS_ACTIVITY_STREAM_TYPE_ACTIVITY_CREATE = 0x0201,
  OS_ACTIVITY_STREAM_TYPE_ACTIVITY_TRANSITION = 0x0202,
  OS_ACTIVITY_STREAM_TYPE_ACTIVITY_USERACTION = 0x0203,

  OS_ACTIVITY_STREAM_TYPE_TRACE_MESSAGE = 0x0300,

  OS_ACTIVITY_STREAM_TYPE_LOG_MESSAGE = 0x0400,
  OS_ACTIVITY_STREAM_TYPE_LEGACY_LOG_MESSAGE = 0x0480,

  OS_ACTIVITY_STREAM_TYPE_SIGNPOST_BEGIN = 0x0601,
  OS_ACTIVITY_STREAM_TYPE_SIGNPOST_END = 0x0602,
  OS_ACTIVITY_STREAM_TYPE_SIGNPOST_EVENT = 0x0603,

  OS_ACTIVITY_STREAM_TYPE_STATEDUMP_EVENT = 0x0A00,
};
typedef uint32_t os_activity_stream_type_t;

enum {
  OS_ACTIVITY_STREAM_EVENT_STARTED = 1,
  OS_ACTIVITY_STREAM_EVENT_STOPPED = 2,
  OS_ACTIVITY_STREAM_EVENT_FAILED = 3,
  OS_ACTIVITY_STREAM_EVENT_CHUNK_STARTED = 4,
  OS_ACTIVITY_STREAM_EVENT_CHUNK_FINISHED = 5,
};
typedef uint32_t os_activity_stream_event_t;

// Types

typedef uint64_t os_activity_id_t;
typedef struct os_activity_stream_s *os_activity_stream_t;
typedef struct os_activity_stream_entry_s *os_activity_stream_entry_t;

#define OS_ACTIVITY_STREAM_COMMON()                                            \
  uint64_t trace_id;                                                           \
  uint64_t timestamp;                                                          \
  uint64_t thread;                                                             \
  const uint8_t *image_uuid;                                                   \
  const char *image_path;                                                      \
  struct timeval tv_gmt;                                                       \
  struct timezone tz;                                                          \
  uint32_t offset

typedef struct os_activity_stream_common_s {
  OS_ACTIVITY_STREAM_COMMON();
} * os_activity_stream_common_t;

struct os_activity_create_s {
  OS_ACTIVITY_STREAM_COMMON();
  const char *name;
  os_activity_id_t creator_aid;
  uint64_t unique_pid;
};

struct os_activity_transition_s {
  OS_ACTIVITY_STREAM_COMMON();
  os_activity_id_t transition_id;
};

typedef struct os_log_message_s {
  OS_ACTIVITY_STREAM_COMMON();
  const char *format;
  const uint8_t *buffer;
  size_t buffer_sz;
  const uint8_t *privdata;
  size_t privdata_sz;
  const char *subsystem;
  const char *category;
  uint32_t oversize_id;
  uint8_t ttl;
  bool persisted;
} * os_log_message_t;

typedef struct os_trace_message_v2_s {
  OS_ACTIVITY_STREAM_COMMON();
  const char *format;
  const void *buffer;
  size_t bufferLen;
  xpc_object_t __unsafe_unretained payload;
} * os_trace_message_v2_t;

typedef struct os_activity_useraction_s {
  OS_ACTIVITY_STREAM_COMMON();
  const char *action;
  bool persisted;
} * os_activity_useraction_t;

typedef struct os_signpost_s {
  OS_ACTIVITY_STREAM_COMMON();
  const char *format;
  const uint8_t *buffer;
  size_t buffer_sz;
  const uint8_t *privdata;
  size_t privdata_sz;
  const char *subsystem;
  const char *category;
  uint64_t duration_nsec;
  uint32_t callstack_depth;
  uint64_t callstack[OS_ACTIVITY_MAX_CALLSTACK];
} * os_signpost_t;

typedef struct os_activity_statedump_s {
  OS_ACTIVITY_STREAM_COMMON();
  char *message;
  size_t message_size;
  char image_path_buffer[PATH_MAX];
} * os_activity_statedump_t;

struct os_activity_stream_entry_s {
  os_activity_stream_type_t type;

  // information about the process streaming the data
  pid_t pid;
  uint64_t proc_id;
  const uint8_t *proc_imageuuid;
  const char *proc_imagepath;

  // the activity associated with this streamed event
  os_activity_id_t activity_id;
  os_activity_id_t parent_id;

  union {
    struct os_activity_stream_common_s common;
    struct os_activity_create_s activity_create;
    struct os_activity_transition_s activity_transition;
    struct os_log_message_s log_message;
    struct os_trace_message_v2_s trace_message;
    struct os_activity_useraction_s useraction;
    struct os_signpost_s signpost;
    struct os_activity_statedump_s statedump;
  };
};

// Blocks

typedef bool (^os_activity_stream_block_t)(os_activity_stream_entry_t entry,
                                           int error);

typedef void (^os_activity_stream_event_block_t)(
    os_activity_stream_t stream, os_activity_stream_event_t event);

// SPI entry point prototypes

typedef os_activity_stream_t (*os_activity_stream_for_pid_t)(
    pid_t pid, os_activity_stream_flag_t flags,
    os_activity_stream_block_t stream_block);

typedef void (*os_activity_stream_resume_t)(os_activity_stream_t stream);

typedef void (*os_activity_stream_cancel_t)(os_activity_stream_t stream);

typedef char *(*os_log_copy_formatted_message_t)(os_log_message_t log_message);

typedef void (*os_activity_stream_set_event_handler_t)(
    os_activity_stream_t stream, os_activity_stream_event_block_t block);

#endif // LLDB_TOOLS_DEBUGSERVER_SOURCE_MACOSX_DARWINLOG_ACTIVITYSTREAMSPI_H
