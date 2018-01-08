/* rtems-task-variable-add.c -- adding a task specific variable in RTEMS OS.

   Copyright 2010 The Go Authors. All rights reserved.
   Use of this source code is governed by a BSD-style
   license that can be found in the LICENSE file.  */

#include <rtems/error.h>
#include <rtems/system.h>
#include <rtems/rtems/tasks.h>

#include "go-assert.h"

/* RTEMS does not support GNU TLS extension __thread.  */
void
__wrap_rtems_task_variable_add (void **var)
{
  rtems_status_code sc = rtems_task_variable_add (RTEMS_SELF, var, NULL);
  if (sc != RTEMS_SUCCESSFUL)
    {
      rtems_error (sc, "rtems_task_variable_add failed");
      __go_assert (0);
    }
}

