#!/usr/bin/env python3

import os
import signal

os.kill(os.getpid(), signal.SIGABRT)
