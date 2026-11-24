# encoding: utf-8

import lldb


class MyFrameRecognizer:
    def get_recognized_arguments(self, frame):
        if frame.name == "foo":
            arg1 = frame.EvaluateExpression("$arg1").signed
            arg2 = frame.EvaluateExpression("$arg2").signed
            val1 = (
                frame.GetThread()
                .GetProcess()
                .GetTarget()
                .CreateValueFromExpression("a", "%d" % arg1)
            )
            val2 = (
                frame.GetThread()
                .GetProcess()
                .GetTarget()
                .CreateValueFromExpression("b", "%d" % arg2)
            )
            return [val1, val2]
        elif frame.name == "bar":
            arg1 = frame.EvaluateExpression("$arg1").signed
            val1 = (
                frame.GetThread()
                .GetProcess()
                .GetTarget()
                .CreateValueFromExpression("a", "(int *)%d" % arg1)
            )
            return [val1]
        return []


class MyOtherFrameRecognizer:
    def get_recognized_arguments(self, frame):
        return []


class BazFrameRecognizer:
    def should_hide(self, frame):
        return "baz" in frame.name


class NestedFrameRecognizer:
    """Exercises the optional hooks beyond `should_hide`: replace the
    stop description, redirect the selected frame, and expose a synthetic
    exception object. Tracks invocation via class-level lists so a test
    can assert each hook actually fires.
    """

    select_most_relevant_frame_calls = []
    get_exception_calls = []
    get_stop_description_calls = []

    def select_most_relevant_frame(self, frame):
        self.select_most_relevant_frame_calls.append(frame.name)
        # Redirect from `nested` (frame 0) to its caller `baz` (frame 1).
        return frame.thread.frames[1] if frame.thread.num_frames > 1 else None

    def get_exception(self, frame):
        self.get_exception_calls.append(frame.name)
        return frame.thread.process.target.CreateValueFromExpression(
            "recognized_exception", "42"
        )

    def get_stop_description(self, frame):
        self.get_stop_description_calls.append(frame.name)
        return "recognized nested()"
