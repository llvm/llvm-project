import lldb
import struct

# The thread this plug-in reports. No thread of the stub backs it.
OS_THREAD_ID = 0x2


class OperatingSystemPlugIn(object):
    def __init__(self, process):
        self.process = process

    def get_thread_info(self):
        return [
            {
                "tid": OS_THREAD_ID,
                "name": "os",
                "state": "stopped",
                "stop_reason": "none",
            }
        ]

    def get_register_info(self):
        return {
            "sets": ["GPR"],
            "registers": [
                {
                    "name": "pc",
                    "bitsize": 64,
                    "offset": 0,
                    "encoding": "uint",
                    "format": "hex",
                    "set": 0,
                    "generic": "pc",
                }
            ],
        }

    def get_register_data(self, tid):
        # A PC outside every module, so the thread executes no instance.
        return struct.pack("Q", 0)
