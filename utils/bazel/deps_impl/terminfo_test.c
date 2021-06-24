/*
This file is licensed under the Apache License v2.0 with LLVM Exceptions.
See https://llvm.org/LICENSE.txt for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
*/

extern int setupterm(char *term, int filedes, int *errret);
extern struct term *set_curterm(struct term *termp);
extern int del_curterm(struct term *termp);
extern int tigetnum(char *capname);

int main() {
  setupterm(0, 0, 0);
  set_curterm(0);
  del_curterm(0);
  tigetnum(0);
}
