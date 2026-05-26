/* ============================================================
   LLVM Advisor — Unit Detail View
   ============================================================ */

const UnitDetailView = {
  async render(params) {
    const unitId = params.id;
    const snapId = params.snapshot || State.get('currentSnapshot')?.id;
    if (!unitId || !snapId) {
      Shell.renderMain(h('div', { class: 'empty-state' }, h('div', {}, 'No unit selected')));
      return;
    }

    const container = h('div', { class: 'unit-detail-layout clean' });
    const sidebar = h('div', { class: 'unit-detail-sidebar' });
    const main = h('div', { class: 'unit-detail-main' });
    container.appendChild(sidebar);
    container.appendChild(main);

    Shell.renderMain(container);

    // Load unit data
    const unitRes = await API.unit(snapId, unitId);
    if (!unitRes.ok) {
      main.appendChild(h('div', { class: 'empty-state' },
        h('div', {}, 'Unit not found'),
        h('div', { class: 'reason' }, unitRes.error)
      ));
      return;
    }
    const unit = unitRes.data;

    const fileInfo = this.pathInfo(unit.source_path || unit.id);
    main.appendChild(h('div', { class: 'unit-detail-hero' },
      h('button', { class: 'back-link', onClick: () => Router.navigate('/units') }, 'Units'),
      h('div', { class: 'unit-title-block' },
        h('div', { class: 'unit-detail-title' }, fileInfo.file || unit.id),
        fileInfo.dir ? h('div', { class: 'unit-detail-path mono', title: unit.source_path || unit.id }, fileInfo.dir) : null
      ),
      h('div', { class: 'unit-detail-meta' },
        [unit.language, unit.target_triple, unit.toolchain_version, timeAgo(unit.created_unix)].filter(Boolean).map(v =>
          h('span', {}, v)
        )
      )
    ));

    const summary = h('div', { class: 'unit-detail-summary', id: 'unit-detail-summary' },
      this.summaryCard('Instructions', unit.instructions, 'neutral'),
      this.summaryCard('Symbols', unit.symbols, 'neutral'),
      this.summaryCard('Warnings', unit.warnings || 0, (unit.warnings || 0) ? 'warn' : 'neutral'),
      this.summaryCard('Errors', unit.errors || 0, (unit.errors || 0) ? 'danger' : 'neutral')
    );
    main.appendChild(summary);

    const tabState = { active: 'Overview', results: [], byCapability: new Map() };

    // Code viewer and tabs
    const tabs = ['Overview', 'Remarks', 'Artifacts'];
    const tabHeaders = h('div', { class: 'code-tabs' });
    const contentArea = h('div', { class: 'code-content', id: 'code-content' });
    const inlineExplorer = h('div', { id: 'inline-explorer' });

    const updateTab = (idx) => {
      tabState.active = tabs[idx];
      Array.from(tabHeaders.children).forEach((t, i) => t.classList.toggle('active', i === idx));
      clearEl(contentArea);
      contentArea.appendChild(this.renderTab(tabs[idx], tabState));
      // Close inline explorer when switching tabs
      clearEl(inlineExplorer);
    };

    tabs.forEach((name, i) => {
      const tab = h('div', { class: 'code-tab' + (i === 0 ? ' active' : ''), onClick: () => updateTab(i) }, name);
      tabHeaders.appendChild(tab);
    });

    const viewer = h('div', { class: 'code-viewer clean-viewer' }, tabHeaders, contentArea);
    main.appendChild(viewer);
    main.appendChild(inlineExplorer);

    this.renderCapSidebar(sidebar, unit);

    // Try to load capability data
    this.loadCapabilities(unit, sidebar, null, tabState, main, () => updateTab(tabs.indexOf(tabState.active)));
  },

  pathInfo(path) {
    const normalized = String(path || '').replace(/\\/g, '/');
    const idx = normalized.lastIndexOf('/');
    return idx >= 0
      ? { dir: normalized.slice(0, idx), file: normalized.slice(idx + 1) }
      : { dir: '', file: normalized };
  },

  summaryCard(label, value, tone) {
    return h('div', { class: `unit-detail-stat ${tone}` },
      h('span', { class: 'unit-detail-stat-value' }, formatNumber(value)),
      h('span', { class: 'unit-detail-stat-label' }, label)
    );
  },

  openFunctionExplorer(unit, snapshotId, functionName) {
    const container = document.getElementById('inline-explorer');
    if (!container) return;
    clearEl(container);

    const card = h('div', { class: 'inline-explorer' });
    const header = h('div', { class: 'inline-explorer-header' },
      h('span', { class: 'capability-card-title mono' }, functionName),
      h('button', {
        class: 'detail-card-close',
        onClick: () => clearEl(container),
        title: 'Close'
      }, '×')
    );

    const controls = h('div', { class: 'cap-pills' });
    const body = h('div', { class: 'capability-stack' });
    const modes = [['remarks', 'Remarks']];

    const loadMode = async (mode, pill) => {
      Array.from(controls.children).forEach(node => node.classList.remove('available'));
      if (pill) pill.classList.add('available');
      clearEl(body);
      body.appendChild(h('div', { class: 'empty-state' },
        h('div', {}, `Loading ${mode}`),
        h('div', { class: 'reason mono' }, functionName)));
      const res = await API.inspect(mode, {
        snapshot_id: snapshotId,
        unit: unit.id,
        function: functionName,
      });
      clearEl(body);
      if (!res.ok) {
        body.appendChild(h('div', { class: 'empty-state' },
          h('div', {}, 'Inspection failed'),
          h('div', { class: 'reason mono' }, res.error || 'unknown error')));
        return;
      }
      body.appendChild(UI.inspectResult(res.data));
    };

    modes.forEach(([mode, label], index) => {
      const pill = h('button', {
        class: `cap-pill ${index === 0 ? 'available' : ''}`,
        onClick: () => loadMode(mode, pill),
      }, label);
      controls.appendChild(pill);
      if (index === 0) setTimeout(() => loadMode(mode, pill), 0);
    });

    card.appendChild(header);
    card.appendChild(controls);
    card.appendChild(body);
    container.appendChild(card);

    // Scroll into view
    card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  },

  renderCapSidebar(sidebar, unit) {
    clearEl(sidebar);
    sidebar.appendChild(h('div', { class: 'unit-side-card' },
      h('div', { class: 'rail-title' }, 'Unit'),
      h('div', { class: 'soft-kv' }, h('span', {}, 'Language'), h('strong', {}, unit.language || '–')),
      h('div', { class: 'soft-kv' }, h('span', {}, 'Target'), h('strong', {}, unit.target_triple || '–')),
      h('div', { class: 'soft-kv' }, h('span', {}, 'Toolchain'), h('strong', {}, unit.toolchain_version || '–'))
    ));
    sidebar.appendChild(h('div', { class: 'unit-side-card', id: 'capability-health-card' },
      h('div', { class: 'rail-title' }, 'Coverage'),
      h('div', { class: 'rail-empty' }, 'Loading analysis coverage')
    ));
    // Function list placeholder in sidebar
  },

  addSection(parent, title, open, kvPairs) {
    const section = h('div', { class: 'cap-section' + (open ? ' open' : '') });
    const header = h('div', { class: 'cap-section-header', onClick: () => section.classList.toggle('open') },
      h('span', {}, title)
    );
    const body = h('div', { class: 'cap-section-body' });
    kvPairs.forEach(([k, v]) => {
      body.appendChild(h('div', { class: 'kv' },
        h('span', { class: 'k' }, k),
        h('span', { class: 'v' }, v)
      ));
    });
    section.appendChild(header);
    section.appendChild(body);
    parent.appendChild(section);
  },

  renderTab(tab, state) {
    if (!state.results.length)
      return h('div', { class: 'empty-state', style: { padding: '40px' } },
        h('div', {}, 'Loading capabilities'),
        h('div', { class: 'reason mono' }, 'Querying analyzer results for this unit'));

    if (tab === 'Remarks') {
      const findings = state.results
        .filter(r => r.capability.includes('remarks'))
        .flatMap(r => r.findings);
      if (!findings.length) {
        return this.emptyTab('No optimization remarks', 'No optimization remarks were reported for this unit.');
      }
      return h('div', { class: 'capability-stack' },
        UI.passTimeline(findings),
        UI.findingList(findings)
      );
    }

    if (tab === 'Artifacts') {
      const artifacts = state.results.flatMap(r => r.artifacts.map(a => ({ capability: r.capability, ...a })));
      if (!artifacts.length)
        return this.emptyTab('No artifacts', 'Capabilities returned metrics, tables, or findings only.');
      return h('div', { class: 'artifact-stack' }, artifacts.map(a => {
        const lang = a.name === 'ir' || a.name === 'llvm' ? 'ir' : (a.name === 'assembly' || a.name === 'asm') ? 'asm' : null;
        return h('div', { class: 'artifact-block' },
          h('div', { class: 'table-title' }, `${friendlyCapabilityName(a.capability)} · ${a.name}`),
          a.text && lang ? Syntax.codeBlock(a.text, lang) : a.text ? h('pre', { class: 'raw-json' }, a.text) : h('pre', { class: 'raw-json' }, JSON.stringify(a.data, null, 2))
        );
      }));
    }

    const metrics = this.collectOverview(state.results);
    return h('div', { class: 'unit-overview-panel' },
      h('div', { class: 'quiet-section-title' }, 'Summary'),
      h('div', { class: 'unit-overview-cards' },
        this.summaryCard('Remarks', metrics.remarks, metrics.remarks ? 'info' : 'neutral')
      ),
      h('div', { class: 'quiet-section-title' }, 'Available Analysis'),
      h('div', { class: 'analysis-list' }, state.results.map(r =>
        h('button', {
          class: `analysis-row ${r.available ? 'available' : 'missing'}`,
          title: r.reason || r.capability,
          onClick: () => Shell.renderDetail(UI.capabilityPanel(r), friendlyCapabilityName(r.capability)),
        },
          h('span', { class: 'analysis-name' }, friendlyCapabilityName(r.capability)),
          h('span', { class: 'analysis-state' }, r.available ? 'Ready' : 'Missing')
        )
      ))
    );
  },

  emptyTab(title, reason) {
    return h('div', { class: 'empty-state soft-empty' },
      h('div', {}, title),
      h('div', { class: 'reason' }, reason));
  },

  collectOverview(results) {
    const metrics = { remarks: 0 };
    results.forEach(r => {
      if (!r.available) return;
      metrics.remarks += Number(r.metrics.count && r.capability.includes('remarks') ? r.metrics.count : 0);
    });
    return metrics;
  },

  async loadCapabilities(unit, sidebar, pills, tabState, main, refresh) {
    const capRes = await API.capabilities();
    const caps = Array.isArray(capRes.data)
      ? capRes.data.filter(spec => CapabilityData.shouldQueryCapability(spec, 'unit')).map(c => c.id).filter(Boolean)
      : ['llvm.remarks.summary', 'llvm.remarks.detail'];
    const res = await API.queryUnit(unit.id, caps);
    if (!res.ok) {
      if (main) main.appendChild(UI.errorCard(res.error || 'query failed', () => this.render({ id: unit.id, snapshot: unit.snapshot_id || State.get('currentSnapshot')?.id })));
      return;
    }

    const results = CapabilityData.normalizeResults(res.data);
    tabState.results = results;
    tabState.byCapability = new Map(results.map(r => [r.capability, r]));
    this.renderCoverage(sidebar, results);

    if (refresh) refresh();
  },

  renderCoverage(sidebar, results) {
    const card = sidebar.querySelector('#capability-health-card');
    if (!card) return;
    clearEl(card);
    const available = results.filter(r => r.available).length;
    card.appendChild(h('div', { class: 'rail-title' }, 'Coverage'));
    card.appendChild(h('div', { class: 'coverage-number' },
      h('strong', {}, `${available}/${results.length}`),
      h('span', {}, ' analyses ready')
    ));
    const list = h('div', { class: 'coverage-list' });
    results.slice(0, 10).forEach(r => list.appendChild(h('div', { class: `coverage-row ${r.available ? 'available' : 'missing'}` },
      h('span', {}, friendlyCapabilityName(r.capability)),
      h('i', {})
    )));
    card.appendChild(list);
  },
};
