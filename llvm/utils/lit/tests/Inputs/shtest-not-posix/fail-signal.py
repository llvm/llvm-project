#!/usr/bin/env python

import os
import signal

os.kill(os.getpid(), signal.SIGABRT)
