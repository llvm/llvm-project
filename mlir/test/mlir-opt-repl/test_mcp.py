"""Tests for the MCP server protocol."""

from conftest import INIT_MSG, SAMPLE_MLIR, parse_responses, run_mcp, tool_call


class TestInitialize:
    def test_response(self):
        output = run_mcp(INIT_MSG)
        r = parse_responses(output)[0]
        assert r["id"] == 0
        assert r["result"]["serverInfo"]["name"] == "mlir-opt-repl"
        assert r["result"]["protocolVersion"] == "2024-11-05"
        assert r["result"]["capabilities"]["tools"]["listChanged"] is False


class TestToolsList:
    def test_all_tools_present(self):
        output = run_mcp(
            INIT_MSG, {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
        )
        tools = parse_responses(output)[1]["result"]["tools"]
        names = {t["name"] for t in tools}
        assert names == {
            "run_pipeline",
            "chain_pipeline",
            "get_current_ir",
            "reset",
            "list_passes",
            "rewind",
            "bookmark",
            "save",
            "verify",
            "history",
        }


class TestRunPipeline:
    def test_basic(self):
        output = run_mcp(
            INIT_MSG,
            tool_call(
                1,
                "run_pipeline",
                {
                    "mlir": SAMPLE_MLIR,
                    "passes": ["convert-arith-to-llvm", "convert-func-to-llvm"],
                },
            ),
        )
        text = parse_responses(output)[1]["result"]["content"][0]["text"]
        assert "llvm.func" in text
        assert "llvm.fadd" in text

    def test_invalid_mlir(self):
        output = run_mcp(
            INIT_MSG,
            tool_call(
                1, "run_pipeline", {"mlir": "not valid", "passes": ["canonicalize"]}
            ),
        )
        r = parse_responses(output)[1]["result"]
        assert r["isError"] is True

    def test_extra_args(self):
        output = run_mcp(
            INIT_MSG,
            tool_call(
                1,
                "run_pipeline",
                {
                    "mlir": "func.func @f() { return }",
                    "passes": ["canonicalize"],
                    "extra_args": ["allow-unregistered-dialect"],
                },
            ),
        )
        r = parse_responses(output)[1]["result"]
        assert "isError" not in r


class TestChainPipeline:
    def test_chain(self):
        output = run_mcp(
            INIT_MSG,
            tool_call(
                1, "run_pipeline", {"mlir": SAMPLE_MLIR, "passes": ["canonicalize"]}
            ),
            tool_call(
                2,
                "chain_pipeline",
                {"passes": ["convert-arith-to-llvm", "convert-func-to-llvm"]},
            ),
        )
        text = parse_responses(output)[2]["result"]["content"][0]["text"]
        assert "llvm.func" in text

    def test_no_state(self):
        output = run_mcp(
            INIT_MSG, tool_call(1, "chain_pipeline", {"passes": ["canonicalize"]})
        )
        r = parse_responses(output)[1]["result"]
        assert r["isError"] is True
        assert "no current IR state" in r["content"][0]["text"]

    def test_error_preserves_state(self):
        output = run_mcp(
            INIT_MSG,
            tool_call(
                1, "run_pipeline", {"mlir": SAMPLE_MLIR, "passes": ["canonicalize"]}
            ),
            tool_call(2, "chain_pipeline", {"passes": ["nonexistent-pass-xyz"]}),
            tool_call(3, "get_current_ir"),
        )
        responses = parse_responses(output)
        assert responses[2]["result"]["isError"] is True
        assert "arith.addf" in responses[3]["result"]["content"][0]["text"]


class TestGetCurrentIR:
    def test_no_state(self):
        output = run_mcp(INIT_MSG, tool_call(1, "get_current_ir"))
        assert (
            "(no IR state set)"
            in parse_responses(output)[1]["result"]["content"][0]["text"]
        )

    def test_with_state(self):
        output = run_mcp(
            INIT_MSG,
            tool_call(
                1,
                "run_pipeline",
                {"mlir": "func.func @f() { return }", "passes": ["canonicalize"]},
            ),
            tool_call(2, "get_current_ir"),
        )
        assert (
            "func.func @f" in parse_responses(output)[2]["result"]["content"][0]["text"]
        )


class TestReset:
    def test_clears_state(self):
        output = run_mcp(
            INIT_MSG,
            tool_call(
                1,
                "run_pipeline",
                {"mlir": "func.func @f() { return }", "passes": ["canonicalize"]},
            ),
            tool_call(2, "reset"),
            tool_call(3, "get_current_ir"),
        )
        responses = parse_responses(output)
        assert "IR state cleared" in responses[2]["result"]["content"][0]["text"]
        assert "(no IR state set)" in responses[3]["result"]["content"][0]["text"]


class TestRewind:
    def test_rewind_one(self):
        output = run_mcp(
            INIT_MSG,
            tool_call(
                1, "run_pipeline", {"mlir": SAMPLE_MLIR, "passes": ["canonicalize"]}
            ),
            tool_call(2, "chain_pipeline", {"passes": ["convert-arith-to-llvm"]}),
            tool_call(3, "rewind", {"steps": 1}),
        )
        text = parse_responses(output)[3]["result"]["content"][0]["text"]
        assert "Rewound 1 step(s)" in text
        assert "arith.addf" in text

    def test_no_history(self):
        output = run_mcp(INIT_MSG, tool_call(1, "rewind", {"steps": 1}))
        r = parse_responses(output)[1]["result"]
        assert r["isError"] is True

    def test_past_beginning(self):
        output = run_mcp(
            INIT_MSG,
            tool_call(
                1,
                "run_pipeline",
                {"mlir": "func.func @f() { return }", "passes": ["canonicalize"]},
            ),
            tool_call(2, "rewind", {"steps": 99}),
        )
        assert (
            "Rewound to beginning"
            in parse_responses(output)[2]["result"]["content"][0]["text"]
        )


class TestHistory:
    def _setup(self):
        return [
            INIT_MSG,
            tool_call(
                1, "run_pipeline", {"mlir": SAMPLE_MLIR, "passes": ["canonicalize"]}
            ),
            tool_call(2, "chain_pipeline", {"passes": ["convert-arith-to-llvm"]}),
        ]

    def test_no_history(self):
        output = run_mcp(INIT_MSG, tool_call(1, "history"))
        assert (
            "(no history)" in parse_responses(output)[1]["result"]["content"][0]["text"]
        )

    def test_unified(self):
        output = run_mcp(*self._setup(), tool_call(3, "history", {"format": "unified"}))
        text = parse_responses(output)[3]["result"]["content"][0]["text"]
        assert "---" in text and "+++" in text

    def test_side_by_side(self):
        output = run_mcp(
            *self._setup(),
            tool_call(3, "history", {"format": "side_by_side", "width": 80}),
        )
        text = parse_responses(output)[3]["result"]["content"][0]["text"]
        assert "--canonicalize" in text and "--convert-arith-to-llvm" in text

    def test_show_ir(self):
        output = run_mcp(*self._setup(), tool_call(3, "history", {"show_ir": True}))
        text = parse_responses(output)[3]["result"]["content"][0]["text"]
        assert "[0] initial" in text and "arith.addf" in text

    def test_pretty_unified(self):
        output = run_mcp(
            *self._setup(),
            tool_call(3, "history", {"format": "unified", "pretty": True}),
        )
        text = parse_responses(output)[3]["result"]["content"][0]["text"]
        assert "\033[" in text

    def test_pretty_side_by_side(self):
        output = run_mcp(
            *self._setup(),
            tool_call(
                3, "history", {"format": "side_by_side", "pretty": True, "width": 80}
            ),
        )
        text = parse_responses(output)[3]["result"]["content"][0]["text"]
        assert "\033[" in text


class TestListPasses:
    def test_filter(self):
        output = run_mcp(
            INIT_MSG, tool_call(1, "list_passes", {"filter": "arith-to-llvm"})
        )
        assert (
            "convert-arith-to-llvm"
            in parse_responses(output)[1]["result"]["content"][0]["text"]
        )

    def test_no_match(self):
        output = run_mcp(
            INIT_MSG, tool_call(1, "list_passes", {"filter": "zzz-nonexistent"})
        )
        assert (
            "(no passes matched)"
            in parse_responses(output)[1]["result"]["content"][0]["text"]
        )


class TestErrors:
    def test_unknown_tool(self):
        output = run_mcp(INIT_MSG, tool_call(1, "unknown_tool"))
        r = parse_responses(output)[1]["result"]
        assert r["isError"] is True
        assert "Unknown tool" in r["content"][0]["text"]

    def test_unknown_method(self):
        output = run_mcp(
            INIT_MSG, {"jsonrpc": "2.0", "id": 1, "method": "bad/method", "params": {}}
        )
        assert "Method not found" in parse_responses(output)[1]["error"]["message"]

    def test_notification_no_response(self):
        output = run_mcp(
            INIT_MSG, {"jsonrpc": "2.0", "method": "notifications/initialized"}
        )
        assert len(parse_responses(output)) == 1


class TestBookmarkMCP:
    def test_bookmark_and_rewind(self):
        output = run_mcp(
            INIT_MSG,
            tool_call(
                1, "run_pipeline", {"mlir": SAMPLE_MLIR, "passes": ["canonicalize"]}
            ),
            tool_call(2, "bookmark", {"name": "pre-lower"}),
            tool_call(3, "chain_pipeline", {"passes": ["convert-arith-to-llvm"]}),
            tool_call(4, "rewind", {"target": "pre-lower"}),
        )
        responses = parse_responses(output)
        assert (
            "Bookmarked [1] as 'pre-lower'"
            in responses[2]["result"]["content"][0]["text"]
        )
        assert (
            "Rewound to bookmark 'pre-lower'"
            in responses[4]["result"]["content"][0]["text"]
        )
        assert "arith.addf" in responses[4]["result"]["content"][0]["text"]

    def test_bookmark_list(self):
        output = run_mcp(
            INIT_MSG,
            tool_call(
                1, "run_pipeline", {"mlir": SAMPLE_MLIR, "passes": ["canonicalize"]}
            ),
            tool_call(2, "bookmark", {"name": "snap"}),
            tool_call(3, "bookmark"),
        )
        responses = parse_responses(output)
        assert "snap" in responses[3]["result"]["content"][0]["text"]

    def test_bookmark_no_history(self):
        output = run_mcp(INIT_MSG, tool_call(1, "bookmark", {"name": "x"}))
        assert parse_responses(output)[1]["result"]["isError"] is True

    def test_bookmark_shown_in_history(self):
        output = run_mcp(
            INIT_MSG,
            tool_call(
                1, "run_pipeline", {"mlir": SAMPLE_MLIR, "passes": ["canonicalize"]}
            ),
            tool_call(2, "bookmark", {"name": "marked"}),
            tool_call(3, "history"),
        )
        assert "marked" in parse_responses(output)[3]["result"]["content"][0]["text"]


class TestSaveMCP:
    def test_save(self, tmp_path):
        path = str(tmp_path / "out.mlir")
        output = run_mcp(
            INIT_MSG,
            tool_call(
                1,
                "run_pipeline",
                {"mlir": "func.func @f() { return }", "passes": ["canonicalize"]},
            ),
            tool_call(2, "save", {"path": path}),
        )
        assert "Saved to" in parse_responses(output)[2]["result"]["content"][0]["text"]
        assert "func.func @f" in open(path).read()

    def test_save_no_ir(self):
        output = run_mcp(INIT_MSG, tool_call(1, "save", {"path": "/tmp/x.mlir"}))
        assert parse_responses(output)[1]["result"]["isError"] is True

    def test_save_no_path(self):
        output = run_mcp(
            INIT_MSG,
            tool_call(
                1,
                "run_pipeline",
                {"mlir": "func.func @f() { return }", "passes": ["canonicalize"]},
            ),
            tool_call(2, "save", {"path": ""}),
        )
        assert parse_responses(output)[2]["result"]["isError"] is True


class TestVerifyMCP:
    def test_valid(self):
        output = run_mcp(
            INIT_MSG,
            tool_call(
                1,
                "run_pipeline",
                {"mlir": "func.func @f() { return }", "passes": ["canonicalize"]},
            ),
            tool_call(2, "verify"),
        )
        assert (
            "IR is valid" in parse_responses(output)[2]["result"]["content"][0]["text"]
        )

    def test_no_ir(self):
        output = run_mcp(INIT_MSG, tool_call(1, "verify"))
        assert parse_responses(output)[1]["result"]["isError"] is True


class TestPassPipelineMCP:
    def test_pipeline_string(self):
        output = run_mcp(
            INIT_MSG,
            tool_call(
                1,
                "run_pipeline",
                {"mlir": SAMPLE_MLIR, "passes": ["builtin.module(canonicalize,cse)"]},
            ),
        )
        text = parse_responses(output)[1]["result"]["content"][0]["text"]
        assert "arith.addf" in text

    def test_chain_pipeline_string(self):
        output = run_mcp(
            INIT_MSG,
            tool_call(
                1, "run_pipeline", {"mlir": SAMPLE_MLIR, "passes": ["canonicalize"]}
            ),
            tool_call(
                2,
                "chain_pipeline",
                {"passes": ["builtin.module(convert-arith-to-llvm)"]},
            ),
        )
        text = parse_responses(output)[2]["result"]["content"][0]["text"]
        assert "llvm.fadd" in text
