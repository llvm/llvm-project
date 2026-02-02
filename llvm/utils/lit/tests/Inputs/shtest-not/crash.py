#!/usr/bin/env python
import os, signal
os.kill(os.getpid(), signal.SIGABRT)
