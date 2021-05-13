## Show that lit reports the path to tools picked up via the use_llvm_tool
## function when the tool is found via an environment variable.

# RUN: %{lit} %{inputs}/use-tool-search-env 2>&1 | \
# RUN:   FileCheck %s -DDIR=%p

# CHECK: note: using test-tool: [[DIR]]{{[\\/]}}Inputs{{[\\/]}}use-tool-search-env{{[\\/]}}test.tool
